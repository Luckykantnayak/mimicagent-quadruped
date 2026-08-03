"""
tools/gen_robot_spec.py

Scaffold a `robots/<name>/robot.yaml` from a MuJoCo model. ~80% of a robot spec is
derivable from the MJCF, so this emits a scaffold you REVIEW and complete rather than
hand-writing 20-40 joints in qpos order.

Auto (exact):        joints (qpos[7:] order + limits), base_body, passive joints,
                     robot_card (DoF, per-joint axis+sign+range, EE reach), paths, defaults.
Auto (# REVIEW):     end_effectors (from `ee_*` sites, else a leaf-body heuristic),
                     base_height_range, camera.
Left for you:        environments, exemplars (they reference scenes/skills you author).

Usage:
    python simulator/tools/gen_robot_spec.py --robot <name> [--mjcf PATH] [--force]

MJCF convention that makes EE detection exact (see CLAUDE.md "Adding a robot"):
put a `<site name="ee_<name>"/>` in each end-effector body.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # simulator/
from utils.scene_compose import compose_model                  # noqa: E402

_AXES = {0: "x", 1: "y", 2: "z"}


def _name(m, objtype, i):
    return mujoco.mj_id2name(m, objtype, i)


def _axis_label(axis):
    j = int(np.argmax(np.abs(axis)))
    return ("-" if axis[j] < 0 else "+") + _AXES[j]


def _body_leaves(m):
    has_child = set(int(p) for p in m.body_parentid[1:])
    return [b for b in range(1, m.nbody) if b not in has_child]


def _lowest_z_of_body(m, d, bid):
    """World z of the lowest CONTACT point of a body's geoms — collidable geoms only
    (contype/conaffinity != 0, so a large visual mesh doesn't inflate the estimate),
    using the geom's real half-extent by type rather than the bounding radius."""
    def bottom(g):
        t = m.geom_type[g]; sz = m.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            half = float(sz[2])
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            half = float(sz[0])
        elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            half = float(sz[0] + sz[1])
        else:
            half = float(m.geom_rbound[g])
        return float(d.geom_xpos[g][2]) - half
    coll = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid
            and (m.geom_contype[g] or m.geom_conaffinity[g])]
    if coll:
        return min(bottom(g) for g in coll)
    any_g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    return min((bottom(g) for g in any_g), default=float(d.xpos[bid][2]))


def scaffold(robot, mjcf):
    m = compose_model(mjcf, "assets/scenes/flat.xml")
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(m, d, kid)
    mujoco.mj_forward(m, d)

    # --- base body (carries the free joint) ---
    base_bid = next((int(m.jnt_bodyid[j]) for j in range(m.njnt)
                     if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE), 1)
    base_body = _name(m, mujoco.mjtObj.mjOBJ_BODY, base_bid)
    base_z = float(d.xpos[base_bid][2])

    # --- actuated vs passive joints, in qpos order ---
    actuated = set()
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            actuated.add(int(m.actuator_trnid[a, 0]))
    hinge_slide = [j for j in range(m.njnt)
                   if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
    joints = sorted(hinge_slide, key=lambda j: int(m.jnt_qposadr[j]))  # qpos[7:] order
    joint_rows, card_joints = [], []
    for j in joints:
        jn = _name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        passive = j not in actuated
        if m.jnt_limited[j]:
            lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            lim = f"limits: [{lo:.4f}, {hi:.4f}]"
        else:
            lo, hi, lim = None, None, "limits: [-3.1416, 3.1416]   # REVIEW (joint has no range)"
        extra = ", passive: true" if passive else ""
        joint_rows.append(f"  - {{name: {jn}, {lim}{extra}}}")
        rng = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "unlimited"
        card_joints.append(f"    - {jn}: axis {_axis_label(m.jnt_axis[j])}, range {rng} rad"
                           + ("  (passive)" if passive else ""))

    # --- end-effectors: prefer ee_* sites, else leaf-body heuristic ---
    ee_sites = [(s, _name(m, mujoco.mjtObj.mjOBJ_SITE, s))
                for s in range(m.nsite)
                if (_name(m, mujoco.mjtObj.mjOBJ_SITE, s) or "").startswith("ee_")]
    ee_rows, card_ees, review_ee = [], [], not ee_sites
    if ee_sites:
        for s, sname in ee_sites:
            bid = int(m.site_bodyid[s]); body = _name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
            ee_name = sname[3:]
            off = max(0.0, float(d.xpos[bid][2]) - _lowest_z_of_body(m, d, bid))
            sole_z = _lowest_z_of_body(m, d, bid)
            free = sole_z > max(0.25, 0.4 * base_z)          # rests high -> not a ground contact
            row = f"  - {{name: {ee_name}, body: {body}, site: {sname}, ground_offset: {off:.3f}"
            row += (", contact: false}" if free else "}")
            ee_rows.append(row)
            card_ees.append(f"    - {ee_name} (body {body})")
    else:
        leaves = _body_leaves(m)
        order = sorted(leaves, key=lambda b: _lowest_z_of_body(m, d, b))
        for bid in order:
            body = _name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
            off = max(0.0, float(d.xpos[bid][2]) - _lowest_z_of_body(m, d, bid))
            free = _lowest_z_of_body(m, d, bid) > max(0.25, 0.4 * base_z)
            row = f"  - {{name: {body}, body: {body}, ground_offset: {off:.3f}"
            row += (", contact: false}" if free else "}")
            ee_rows.append(row)
            card_ees.append(f"    - {body}")

    # --- REVIEW-marked estimates ---
    bhr = f"[{0.35 * base_z:.2f}, {2.5 * base_z:.2f}]"
    extent = float(np.linalg.norm(d.xpos[:, :3].max(0) - d.xpos[:, :3].min(0)))
    cam = (f"{{lookat: [0.0, 0.0, {base_z:.2f}], distance: {2.5 * extent:.1f}, "
           f"azimuth: 90, elevation: -15}}")
    dof = len(joints)

    # --- assemble ---
    model_rel = os.path.relpath(mjcf, ".")
    L = [
        f"# robots/{robot}/robot.yaml  — SCAFFOLD generated by gen_robot_spec.py.",
        f"# Review every '# REVIEW' line, then: write example.py, run gen_robot_summary.",
        "",
        f"name: {robot}",
        "scene_xml: assets/scenes/flat.xml",
        f"mjcf: {model_rel}",
        f"base_body: {base_body}",
        "sim_freq: 200",
        f"null_space_gain: {0.02 if dof > 20 else 0.1}",
        "",
        "# End-effectors (ordered). " + (
            "From ee_* sites." if not review_ee
            else "# REVIEW: no ee_* sites found — these are a LEAF-BODY GUESS; keep the real"
                 " contacts, drop the rest, add ee_* sites to the model for an exact spec."),
        "# contact: false = an IK target NOT treated as a ground contact (e.g. a hand).",
        "end_effectors:",
        *ee_rows,
        "",
        "# In model qpos[7:] order (what gets recorded).",
        "joints:",
        *joint_rows,
        "",
        f"base_height_range: {bhr}   # REVIEW (derived from home base z {base_z:.2f} m)",
        f"camera: {cam}   # REVIEW",
        "",
        f"example_skill: robots/{robot}/example.py",
        "robot_card: |",
        f"  {dof}-DoF robot. Base body `{base_body}` stands ~{base_z:.2f} m at the home pose.",
        "  End-effectors:",
        *[f"  {c}" for c in card_ees],
        "  Actuated joints (axis sign, range):",
        *[f"  {c}" for c in card_joints],
        "",
        "# Named scenes this robot's skills may run in (add as you author scenes under",
        "# assets/scenes/). A 'flat' default is always implied.",
        "environments: []",
        "",
        "# Curated reference skills for in-context learning (add 2-3 as you author them",
        "# under robots/<name>/exemplars/ and want the coder to learn from them).",
        "exemplars: []",
    ]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--mjcf", default=None, help="model path (default: the one .xml under assets/robots/<name>/)")
    ap.add_argument("--out", default=None, help="output path (default: robots/<name>/robot.yaml)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing robot.yaml")
    args = ap.parse_args()

    mjcf = args.mjcf
    if not mjcf:
        cand = sorted(p for p in Path(f"assets/robots/{args.robot}").glob("*.xml"))
        if not cand:
            sys.exit(f"No model .xml under assets/robots/{args.robot}/ — pass --mjcf.")
        mjcf = str(cand[0])

    out = args.out or f"robots/{args.robot}/robot.yaml"
    if os.path.exists(out) and not args.force and not args.out:
        sys.exit(f"{out} exists — pass --force to overwrite (or --out to a temp path to inspect).")

    text = scaffold(args.robot, mjcf)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text)
    print(f"[spec] wrote {out}  (from {mjcf})")
    print("[spec] NEXT: review the '# REVIEW' lines, write robots/{}/example.py, "
          "then `python simulator/tools/gen_robot_summary.py --robot {}`.".format(args.robot, args.robot))


if __name__ == "__main__":
    main()
