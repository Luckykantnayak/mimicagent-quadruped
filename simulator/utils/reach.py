"""
simulator/utils/reach.py

Shared limb-reach geometry — the SINGLE source of truth for the base<->contact
reachability law, used by BOTH the runtime base-reach seat
(core/controller.py:_reach_seat_base) and the pre-sim feasibility lint
(tools/reach_lint.py). Keeping the math in one place means the seat that CORRECTS
a base at sim time and the lint that WARNS about it before the sim can never
disagree about what "reachable" means.

Pure geometry: given a base pose, a declared world contact, and a limb's reach +
root offset (from the RobotSpec via the controller's _compute_limb_geom), it
answers "is this contact within the limb's reach, and if not, what base height
would put it back in reach?" No MuJoCo, no IK, no dynamics.
"""
import numpy as np
import mujoco
from utils.math_utils import quat_to_rotation_matrix


def compute_limb_geometry(model, data, base_body, ee_names, ee_bodies):
    """Per-EE {reach, root_offset, cur_dist} at the pose `data` is CURRENTLY in
    (caller must have run forward kinematics). reach = sum of segment lengths from
    the limb's own root (the child of the last branch point on the chain to the
    base) down to the EE = the limb's full extension; root_offset = that root's
    BODY-frame offset from the base; cur_dist = the EE-to-root distance at this
    pose (so cur_dist/reach is the current % extension). Pure kinematics — the ONE
    definition of "reach" shared by the runtime seat, the pre-sim lint, and the
    onboarding singularity check, so they can never disagree."""
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
    Rb = data.xmat[base_bid].reshape(3, 3)
    base_pos = data.xpos[base_bid]
    parent = model.body_parentid
    child_counts = np.bincount(parent, minlength=model.nbody)
    geom = {}
    for name, body in zip(ee_names, ee_bodies):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        chain, b = [], bid
        while b != base_bid and b > 0:
            chain.append(b)
            b = int(parent[b])
        chain = list(reversed(chain))
        if not chain:
            continue
        root_b = chain[0]
        for cb in chain:                       # last body whose parent branches
            if child_counts[parent[cb]] > 1:
                root_b = cb
        ri = chain.index(root_b)
        reach = sum(float(np.linalg.norm(model.body_pos[cb])) for cb in chain[ri + 1:])
        root_off = Rb.T @ (data.xpos[root_b] - base_pos)
        cur = float(np.linalg.norm(data.xpos[bid] - data.xpos[root_b]))
        geom[name] = {"reach": float(reach),
                      "root_offset": np.asarray(root_off, dtype=float),
                      "cur_dist": cur}
    return geom


def contact_reach(base_pos, base_quat, contact_world, limb_geom, target_margin=0.98):
    """Reachability of ONE declared world contact from the base.

    Returns (over, z_lo, z_hi):
      over : ||contact - limb_root|| / full_reach at the CURRENT base
             (>1 => the contact is beyond the limb's full extension => it FLOATS).
      z_lo, z_hi : the base-Z interval that keeps the contact within
             target_margin*reach (a slight bend the IK can hit). Move the base
             into [z_lo, z_hi] and the contact becomes reachable. Both are np.nan
             when the contact is beyond the limb's HORIZONTAL reach — no base
             height fixes it (the base xy/orientation is wrong; that is the base
             solve, not a z seat).

    limb_geom = {"reach": float, "root_offset": (3,) body-frame offset of the
    limb root from the base}. Approximate: ignores torso joints between the base
    and the limb root (parity with how a skill authors a fixed shoulder/hip offset).
    """
    reach = float(limb_geom.get("reach", 0.0))
    if reach <= 1e-6:
        return 0.0, np.nan, np.nan
    R = quat_to_rotation_matrix(np.asarray(base_quat, dtype=float))
    r0 = R @ np.asarray(limb_geom["root_offset"], dtype=float)   # limb root, world offset
    root = np.asarray(base_pos, dtype=float) + r0
    c = np.asarray(contact_world, dtype=float)
    over = float(np.linalg.norm(c - root)) / reach
    reff = target_margin * reach
    dh = float(np.linalg.norm(c[:2] - root[:2]))
    if dh >= reff:                                    # beyond horizontal reach
        return over, np.nan, np.nan
    dz = float(np.sqrt(reff * reff - dh * dh))
    r0z = float(r0[2])
    return over, c[2] - dz - r0z, c[2] + dz - r0z


def base_z_feasible_interval(base_pos, base_quat, contacts_world, limb_geom,
                             target_margin=0.98):
    """Intersect the per-contact base-Z intervals over a set of declared contacts.
    Returns (lo, hi, over, horiz) where [lo, hi] is the base-Z band that keeps
    EVERY (horizontally-reachable) contact in reach (lo>hi => the contacts want
    incompatible heights), `over` is the worst over-reach ratio across contacts,
    and `horiz` lists the contacts that are beyond horizontal reach (z can't fix).
    `contacts_world` = {name: world_pos(3)}; limb_geom = {name: {reach, root_offset}}."""
    lo, hi = -np.inf, np.inf
    over = 0.0
    horiz = []
    for name, c in contacts_world.items():
        g = limb_geom.get(name)
        if not g or c is None:
            continue
        o, zlo, zhi = contact_reach(base_pos, base_quat, c, g, target_margin)
        over = max(over, o)
        if np.isnan(zlo):
            horiz.append(name)
            continue
        lo = max(lo, zlo)
        hi = min(hi, zhi)
    return lo, hi, over, horiz
