"""
simulator/utils/robot_dynamics.py

Physics-grounded feasibility limits distilled from a robot's MuJoCo model.
Used by simulator/tools/gen_robot_summary.py to build {robot_summary}, and
importable anywhere a RobotSpec + MjModel/MjData (reset to the "home" keyframe)
are available.

Two tiers, matching the data actually present in the MJCF:

  KINEMATIC (always available)
    - per-joint position limits (from the spec)

  DYNAMIC (only when the MJCF carries mass/inertia AND actuator torque limits)
    - per-joint acceleration ceiling         q̈_max = τ_max / I_eff        (Newton–Euler, τ = I·q̈)
    - per-joint peak velocity (oscillatory)  q̇_peak = √(A · τ_max / I)    (sinusoid at the accel ceiling)
    - torso forward speed                     v ≈ √(g·L)                    (Froude number, Fr = v²/gL)
    - torso vertical jump / flight time       h = v_z²/2g,  t_air = 2 v_z/g (projectile motion)
    - torso roll/pitch/yaw rates              I·ω = const (aerial), τ = I·α (grounded)

Everything derived from torque is a *ceiling / order-of-magnitude* estimate — a
kinematic simulator applies no forces, so these numbers are guidance for the
planner/coder to keep commanded velocities physically plausible, not hard
constraints enforced by the sim.

Grounding / sources (equations are standard mechanics; the locomotion-specific
relations are cited so the numbers are not invented):
  - Froude number & max walking speed v=√(gL) at Fr=1 (walk→run at Fr≈0.5):
    R. McN. Alexander; https://en.wikipedia.org/wiki/Froude_number
  - Projectile flight time / apex height: standard kinematics.
  - Aerial rotation by conservation of angular momentum (I·ω=const), and that a
    ~9 kg quadruped can generate a full 360° backflip: MIT Mini Cheetah,
    Katz/Di Carlo/Kim, ICRA 2019; https://news.mit.edu/2019/mit-mini-cheetah-first-four-legged-robot-to-backflip-0304
  - Real Unitree Go2 top speed 2.5–5 m/s (sanity check on the Froude estimate):
    https://www.unitree.com/go2/
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import mujoco

G = 9.80665  # m/s^2


# ---------------------------------------------------------------------------
# small MuJoCo helpers
# ---------------------------------------------------------------------------
def _bid(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _jid(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _chain_to_base(model, ee_body: str, base_body: str) -> List[int]:
    """Body ids from base_body (exclusive) down to ee_body (inclusive)."""
    base = _bid(model, base_body)
    chain, b = [], _bid(model, ee_body)
    while b != base and b > 0:
        chain.append(b)
        b = model.body_parentid[b]
    return list(reversed(chain))


def free_base_rot_dofs(model) -> Optional[List[int]]:
    """DOF indices for the floating base's rotation (roll,pitch,yaw), or None.
    A free joint contributes 6 DOFs [vx,vy,vz, wx,wy,wz]; the last three are the
    body-frame angular velocity about x/y/z."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            d = model.jnt_dofadr[j]
            return [d + 3, d + 4, d + 5]
    return None


# ---------------------------------------------------------------------------
# inertia & gravity (need MjData already at the home keyframe + mj_forward)
# ---------------------------------------------------------------------------
def joint_space_inertia(model, data) -> np.ndarray:
    """Diagonal of the full joint-space inertia matrix M(q) at the current pose
    — the effective inertia 'seen' by each DOF. Shape (nv,)."""
    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, M)        # mujoco >= 3.x signature (m, d, dst)
    except TypeError:
        mujoco.mj_fullM(model, M, data.qM)     # older signature (m, dst, qM)
    return np.diag(M).copy()


def gravity_hold_torque(model, data) -> np.ndarray:
    """qfrc_bias per DOF: the generalised force needed to hold the pose against
    gravity (Coriolis ≈ 0 at rest). Used as a sanity gate — a stated torque
    limit below this can't even support the robot, so the datum is unreliable."""
    return data.qfrc_bias.copy()


# ---------------------------------------------------------------------------
# torque limits — prefer actuatorfrcrange (the repo's XML standard), fall back
# to the actuator (motor gear × ctrlrange, or forcerange)
# ---------------------------------------------------------------------------
def torque_limits(model, spec) -> Dict[str, Tuple[Optional[float], str]]:
    """{joint_name: (tau_Nm | None, source)} for actuated joints.
    source ∈ {'actuatorfrcrange', 'actuator', 'none'}."""
    # map joint id -> actuator-side torque ceiling
    act: Dict[int, float] = {}
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jid = int(model.actuator_trnid[a, 0])
        gear = float(model.actuator_gear[a, 0])
        if model.actuator_forcelimited[a]:
            t = float(np.max(np.abs(model.actuator_forcerange[a])))
        elif model.actuator_ctrllimited[a]:
            t = abs(gear) * float(np.max(np.abs(model.actuator_ctrlrange[a])))
        else:
            continue
        act[jid] = max(act.get(jid, 0.0), t)

    out: Dict[str, Tuple[Optional[float], str]] = {}
    for jn in spec.actuated_joint_names:
        jid = _jid(model, jn)
        if jid < 0:
            out[jn] = (None, "none")
            continue
        if model.jnt_actfrclimited[jid]:                       # our standard
            lo, hi = model.jnt_actfrcrange[jid]
            out[jn] = (float(max(abs(lo), abs(hi))), "actuatorfrcrange")
        elif jid in act:
            out[jn] = (act[jid], "actuator")
        else:
            out[jn] = (None, "none")
    return out


def has_usable_dynamics(model, spec) -> bool:
    """True when mass is present and at least one actuated joint has a torque
    limit — the gate for emitting the dynamics tier."""
    if model.body_mass.sum() <= 0:
        return False
    return any(t is not None for t, _ in torque_limits(model, spec).values())


# ---------------------------------------------------------------------------
# per-joint acceleration & velocity ceilings
# ---------------------------------------------------------------------------
def joint_accel_velocity_limits(model, data, spec) -> Dict[str, Dict]:
    """Per actuated joint:
        q̈_max = τ_max / I_eff                         (τ = I·q̈)
        q̇_peak = √(A · q̈_max)  with A = half the joint range
                 — peak speed of a full-range sinusoid ridden at the accel
                   ceiling: for q=A·sin(ωt), q̈_max=A·ω² ⇒ q̇_peak=A·ω=√(A·q̈_max)
    'reliable' is False when τ_max < gravity-hold torque (bad model datum)."""
    inertia = joint_space_inertia(model, data)
    grav = np.abs(gravity_hold_torque(model, data))
    taus = torque_limits(model, spec)
    limits = spec.joint_limits

    out: Dict[str, Dict] = {}
    for jn in spec.actuated_joint_names:
        jid = _jid(model, jn)
        tau, source = taus.get(jn, (None, "none"))
        rec: Dict = {"tau": tau, "source": source, "inertia": None,
                     "qddot_max": None, "qdot_peak": None, "amp": None,
                     "reliable": tau is not None}
        if jid >= 0:
            dof = int(model.jnt_dofadr[jid])
            I = float(inertia[dof])
            rec["inertia"] = I
            if tau is not None and tau < grav[dof]:
                rec["reliable"] = False          # can't even hold against gravity
            if tau is not None and I > 1e-9:
                # Torque available for ACCELERATION = stall torque minus the
                # gravity-hold torque at this pose — the actuator must supply BOTH
                # g and I·q̈ at once. This removes the estimator's one systematic
                # optimism (per a physics review); everything else (off-diagonal
                # inertia coupling, single-pose M, sinusoid-vs-bang-bang) nets out
                # within the check's 1.5× slack. Floored at 10% of stall so a
                # gravity-marginal joint keeps a sane, non-zero ceiling.
                tau_acc = max(float(tau) - float(grav[dof]), 0.1 * float(tau))
                qdd = tau_acc / I
                rec["qddot_max"] = qdd
                if jn in limits:
                    lo, hi = limits[jn]
                    A = 0.5 * abs(hi - lo)
                    rec["amp"] = A
                    rec["qdot_peak"] = float(np.sqrt(max(A, 0.0) * qdd))
        out[jn] = rec
    return out


# ---------------------------------------------------------------------------
# leg geometry (for the torso estimates) — generic, from the contact EEs
# ---------------------------------------------------------------------------
def _leg_summary(model, data, spec) -> Dict:
    """Per-robot leg geometry from the contact end-effectors: standing leg
    length, number of support legs, longest lower-limb segment, the strongest
    torque along a leg, and the horizontal foot offsets from the base."""
    base = _bid(model, spec.base_body)
    base_p = data.xpos[base]
    taus = torque_limits(model, spec)

    n_legs, seg_lens, leg_torques, offs = 0, [], [], []
    for i, (name, body) in enumerate(zip(spec.ee_names, spec.ee_bodies)):
        if i not in spec.contact_ee_indices:
            continue
        n_legs += 1
        chain = _chain_to_base(model, body, spec.base_body)
        if chain:
            seg_lens.append(max(float(np.linalg.norm(model.body_pos[b])) for b in chain))
        # strongest actuated torque along this leg
        chain_set = set(chain)
        best = 0.0
        for jn in spec.actuated_joint_names:
            jid = _jid(model, jn)
            if jid >= 0 and int(model.jnt_bodyid[jid]) in chain_set:
                t, _ = taus.get(jn, (None, ""))
                if t:
                    best = max(best, t)
        if best > 0:
            leg_torques.append(best)
        offs.append(data.xpos[_bid(model, body)] - base_p)

    offs = np.array(offs) if offs else np.zeros((1, 3))
    return {
        "leg_length": float(base_p[2]),                       # standing hip height ≈ leg length
        "n_support": max(n_legs, 1),
        "lower_seg": float(np.median(seg_lens)) if seg_lens else 0.0,
        "leg_torque": float(np.median(leg_torques)) if leg_torques else 0.0,
        "x_offset": float(np.max(np.abs(offs[:, 0]))),        # fore-aft foot spread (pitch lever)
        "y_offset": float(np.max(np.abs(offs[:, 1]))),        # lateral foot spread (roll lever)
        "radial": float(np.max(np.linalg.norm(offs[:, :2], axis=1))),  # yaw lever
    }


# ---------------------------------------------------------------------------
# torso forward speed — Froude number
# ---------------------------------------------------------------------------
def base_linear_speed(model, data, spec) -> Dict:
    """Walking speed ceiling from the Froude number Fr = v²/(g·L) with L = leg
    length (standing hip height): a walking gait keeps a foot on the ground only
    while Fr ≲ 1, so v_walk_max = √(g·L); the walk→run transition is near Fr≈0.5.
    Dynamic running gaits exceed this (real quadrupeds gallop at Fr≈2–4)."""
    L = _leg_summary(model, data, spec)["leg_length"]
    v_walk = float(np.sqrt(G * L)) if L > 0 else 0.0
    return {
        "leg_length": L,
        "v_transition": np.sqrt(0.5) * v_walk,   # Fr = 0.5
        "v_walk": v_walk,                        # Fr = 1.0
        "v_run": 2.0 * v_walk,                   # Fr ≈ 4, dynamic running (indicative)
    }


# ---------------------------------------------------------------------------
# torso vertical jump & aerial rotation
# ---------------------------------------------------------------------------
def base_vertical_jump(model, data, spec) -> Dict:
    """Approximate jump from a leg push, then projectile flight.
      per-leg vertical foot force  F ≈ τ_leg / ℓ_lower        (statics, τ = F·ℓ)
      net upward accel             a_z = n·F/m − g
      take-off speed               v_z = √(2·a_z·Δs),  Δs ≈ 0.3·L  (crouch stroke)
      apex height / flight time    h = v_z²/2g,  t_air = 2·v_z/g   (projectile motion)
    Force term is a rough statics estimate; treat h/t_air as order-of-magnitude."""
    leg = _leg_summary(model, data, spec)
    m = float(model.body_mass.sum())
    L, ell, tau, n = leg["leg_length"], leg["lower_seg"], leg["leg_torque"], leg["n_support"]
    out = {"mass": m, "leg_length": L, "a_z": None, "v_takeoff": None,
           "jump_height": None, "flight_time": None, "flip_rate": None}
    if ell > 1e-3 and tau > 0 and m > 0:
        F = tau / ell
        a_z = n * F / m - G
        out["a_z"] = a_z
        if a_z > 0:
            dz = 0.3 * L
            v_z = float(np.sqrt(2 * a_z * dz))
            t_air = 2 * v_z / G
            out.update(v_takeoff=v_z, jump_height=v_z ** 2 / (2 * G),
                       flight_time=t_air,
                       flip_rate=(2 * np.pi / t_air) if t_air > 0 else None)
    return out


def base_angular_rates(model, data, spec) -> Dict:
    """Torso rotation limits.
      inertia   : whole-body rotational inertia about roll/pitch/yaw (exact,
                  diagonal of M for the free joint's angular DOFs).
      grounded  : with a foot planted a leg force F=τ/ℓ at lever d makes torque
                  τ_b=n·F·d, so α_max=τ_b/I and, over a swing Δθ, ω≈√(α·Δθ).
      aerial    : in flight I·ω is conserved; a rotation needs ω≥2π/t_air, the
                  legs supplying angular impulse L=I·ω at take-off (feasible for a
                  ~9 kg quadruped — MIT Mini Cheetah lands 360° backflips)."""
    rot = free_base_rot_dofs(model)
    leg = _leg_summary(model, data, spec)
    m = float(model.body_mass.sum())
    inertia = joint_space_inertia(model, data)

    axes = ["roll", "pitch", "yaw"]
    levers = {"roll": leg["y_offset"], "pitch": leg["x_offset"], "yaw": leg["radial"]}
    F = (leg["leg_torque"] / leg["lower_seg"]) if leg["lower_seg"] > 1e-3 else 0.0

    out: Dict[str, Dict] = {}
    for k, ax in enumerate(axes):
        I = float(inertia[rot[k]]) if rot else None
        rec = {"inertia": I, "alpha_max": None, "omega_peak": None}
        # a planted-foot lever below ~2 cm (e.g. a biped's feet under its CoM)
        # gives no grounded authority about that axis — such rotation is aerial.
        if I and I > 1e-6 and F > 0 and levers[ax] > 0.02:
            tau_b = leg["n_support"] * F * levers[ax]
            alpha = tau_b / I
            rec["alpha_max"] = alpha
            rec["omega_peak"] = float(np.sqrt(alpha * (np.pi / 2)))  # over a 90° swing
        out[ax] = rec
    out["mass"] = m
    return out
