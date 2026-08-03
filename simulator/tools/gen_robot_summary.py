"""
Generate a compact physical summary for a robot, distilled from its MuJoCo
model, and write it to robots/<name>/summary.md. The prompt loader injects the
result as {robot_summary} so the planner/coder keep motions physically feasible.

Run ONCE when a robot is added:
    python simulator/tools/gen_robot_summary.py --robot <name>

Two tiers are produced:
  - KINEMATICS  (always): standing height, EE reach, limb lengths, and the
    per-joint POSITION limits.
  - DYNAMICS    (auto — emitted when the MJCF carries mass/inertia AND actuator
    torque limits): per-joint acceleration & velocity ceilings, plus torso
    forward-speed, jump, and roll/pitch/yaw-rate estimates. All physics is
    grounded/cited in simulator/utils/robot_dynamics.py. Use --no-dynamics to
    force kinematics-only, or --dynamics to attempt it even if the gate fails.

XML FORMAT STANDARD (so this extracts cleanly for a new robot): give each
actuated joint a `range` (position limits) and an `actuatorfrcrange` (torque
limit, N·m) — the preferred, unambiguous torque source — and let mass/inertia
come from <inertial> tags or geom density. Without a torque source the dynamics
tier is skipped and the agent picks velocities freely.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "simulator"))
from utils.robot_config import load_robot_spec        # noqa: E402
import utils.robot_dynamics as rd                      # noqa: E402


def _bid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def _bname(m, i):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)


def _limb_reach(m, chain):
    """Reach of the end-effector from its limb root (child of the last branch
    point on the chain); reach = sum of the segment lengths below that root."""
    child_counts = np.bincount(m.body_parentid, minlength=m.nbody)
    root_idx = 0
    for i, b in enumerate(chain):
        if child_counts[m.body_parentid[b]] > 1:
            root_idx = i
    return sum(float(np.linalg.norm(m.body_pos[b])) for b in chain[root_idx + 1:])


def check_home_singularity(m, spec, kid):
    """FUNCTIONAL IK-conditioning probe at the home pose. A home that puts a leg at
    (near) full extension is an IK SINGULARITY: the knee has no bend to work with, so
    when a skill later drives the foot toward a foothold the IK folds the limb the
    WRONG way and the contact FLOATS — the exact failure a straight-leg model import
    (e.g. from Menagerie) shows, and one a STATIC extension-% check cannot flag,
    because some fully-extended homes work fine. So probe FUNCTIONALLY with the REAL
    IK: command each declared CONTACT limb to a reachable target 70% of its reach
    straight DOWN from its own root and check it can get there. A limb that cannot
    reach within HALF its own reach has folded/stuck. Relative threshold ->
    morphology-agnostic. Returns [(ee_name, residual_m, reach_m), ...] of folded limbs."""
    from utils.kinematics import MultiLinkGradientDescentIK
    from utils.reach import compute_limb_geometry
    d = mujoco.MjData(m)

    def to_home():
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_kinematics(m, d)

    to_home()
    ik = MultiLinkGradientDescentIK(m, d, spec.ee_bodies, sites=spec.ee_sites,
                                    joint_names=spec.actuated_joint_names,
                                    null_space_gain=spec.null_space_gain)
    geom = compute_limb_geometry(m, d, spec.base_body, spec.ee_names, spec.ee_bodies)
    base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, spec.base_body)
    folded = []
    for i in spec.contact_ee_indices:
        n = spec.ee_names[i]
        g = geom.get(n)
        if not g or g["reach"] < 1e-6:
            continue
        to_home()
        R = d.xmat[base_bid].reshape(3, 3)
        root = d.xpos[base_bid] + R @ g["root_offset"]
        tgt = root.copy(); tgt[2] -= 0.70 * g["reach"]      # 70% reach straight down (reachable)
        cur = ik.get_ee_positions().copy(); mask = np.zeros(len(spec.ee_names), bool)
        cur[i] = tgt; mask[i] = True
        ik.calculate(cur, active_mask=mask, max_iter=250)
        res = float(np.linalg.norm(ik.get_ee_positions()[i] - tgt))
        if res > 0.5 * g["reach"]:
            folded.append((n, res, float(g["reach"])))
    return folded


# ---------------------------------------------------------------------------
# KINEMATICS
# ---------------------------------------------------------------------------
def kinematic_section(m, d, spec):
    L = ["## Kinematics", ""]
    L.append(f"- Base body `{spec.base_body}` nominal standing height: "
             f"{d.xpos[_bid(m, spec.base_body)][2]:.2f} m")
    L.append(f"- Total height (tallest point at rest): {d.xpos[:, 2].max():.2f} m")
    L.append("")
    L.append("- End-effector reach (how far each EE extends from its limb root; "
             "keep commanded targets within this):")
    for name, body in zip(spec.ee_names, spec.ee_bodies):
        chain = rd._chain_to_base(m, body, spec.base_body)
        reach = _limb_reach(m, chain)
        nom = d.xpos[_bid(m, body)] - d.xpos[_bid(m, spec.base_body)]
        L.append(f"    - {name}: reach ≈ {reach:.2f} m; nominal position (rel. base) "
                 f"≈ [{nom[0]:+.2f}, {nom[1]:+.2f}, {nom[2]:+.2f}] m")
    L.append("")
    L.append("- Joint position limits (radians; degrees in parentheses):")
    for jn, (lo, hi) in spec.joint_limits.items():
        L.append(f"    - {jn}: [{lo:+.3f}, {hi:+.3f}] rad "
                 f"([{np.degrees(lo):+.0f}°, {np.degrees(hi):+.0f}°])")
    if spec.passive_joint_indices:
        passive = [spec.joint_names[i] for i in spec.passive_joint_indices]
        L.append(f"    - passive (unactuated): {', '.join(passive)}")
    L.append("")
    L.append("- Limb segment lengths:")
    for b in range(1, m.nbody):
        seg = float(np.linalg.norm(m.body_pos[b]))
        if seg > 1e-4:
            L.append(f"    - {_bname(m, b)}: {seg*100:.1f} cm")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# DYNAMICS
# ---------------------------------------------------------------------------
def dynamics_section(m, d, spec):
    L = ["## Dynamics (feasible velocities & accelerations)", ""]
    L.append("Estimates grounded in standard mechanics (the sim itself applies no "
             "forces — these bound what is *physically plausible*, so keep commanded "
             "velocities within them). Sources cited at the end.")
    L.append("")
    L.append(f"- Total mass: {m.body_mass.sum():.1f} kg")

    # ---- per-joint accel / velocity ----
    jl = rd.joint_accel_velocity_limits(m, d, spec)
    src = next((r["source"] for r in jl.values() if r["tau"] is not None), "none")
    L.append(f"- Per actuated joint — torque limit (source: {src}), effective inertia "
             "at rest, max acceleration |q̈| ≲ τ/I, and peak velocity |q̇| ≲ √(A·τ/I) "
             "for a full-range oscillation:")
    for jn, r in jl.items():
        if r["tau"] is None:
            L.append(f"    - {jn}: no torque limit in model → choose freely")
            continue
        line = (f"    - {jn}: τ ≤ {r['tau']:.0f} N·m, I ≈ {r['inertia']:.3f} kg·m²")
        if r["qddot_max"] is not None:
            line += f" → |q̈| ≲ {r['qddot_max']:.0f} rad/s², |q̇| ≲ {r['qdot_peak']:.1f} rad/s"
        if not r["reliable"]:
            line += "  [⚠ below gravity-hold torque — datum unreliable]"
        L.append(line)
    L.append("")

    # ---- torso forward speed (Froude) ----
    lin = rd.base_linear_speed(m, d, spec)
    L.append("- Torso forward speed (Froude number Fr = v²/gL, L = leg length "
             f"{lin['leg_length']:.2f} m):")
    L.append(f"    - walk→run transition (Fr≈0.5): ≈ {lin['v_transition']:.2f} m/s")
    L.append(f"    - walking ceiling (Fr=1, foot leaves ground above this): ≈ {lin['v_walk']:.2f} m/s")
    L.append(f"    - dynamic running gait (Fr≈4): up to ≈ {lin['v_run']:.2f} m/s")
    L.append("")

    # ---- torso jump / aerial ----
    j = rd.base_vertical_jump(m, d, spec)
    L.append("- Torso vertical jump & flight (leg push F≈τ/ℓ, then projectile "
             "h=v_z²/2g, t_air=2v_z/g — approximate):")
    if j["v_takeoff"] is not None:
        L.append(f"    - net upward accel ≈ {j['a_z']:.0f} m/s², take-off speed ≈ {j['v_takeoff']:.2f} m/s")
        L.append(f"    - apex height ≈ {j['jump_height']:.2f} m, flight time ≈ {j['flight_time']:.2f} s")
        L.append(f"    - one full aerial rotation needs ω ≈ 2π/t_air ≈ {j['flip_rate']:.1f} rad/s "
                 f"({np.degrees(j['flip_rate']):.0f}°/s)")
    else:
        L.append("    - insufficient torque/geometry to estimate a jump")
    L.append("")

    # ---- torso roll/pitch/yaw ----
    ang = rd.base_angular_rates(m, d, spec)
    L.append("- Torso angular rates about roll(x)/pitch(y)/yaw(z) — whole-body "
             "rotational inertia (exact), grounded α=τ/I with a planted foot, and "
             "the aerial regime where I·ω is conserved:")
    for ax in ["roll", "pitch", "yaw"]:
        r = ang[ax]
        if r["inertia"] is None:
            continue
        line = f"    - {ax}: I ≈ {r['inertia']:.2f} kg·m²"
        if r["alpha_max"] is not None:
            line += (f"; grounded |α| ≲ {r['alpha_max']:.0f} rad/s², "
                     f"|ω| ≲ {r['omega_peak']:.1f} rad/s over a 90° swing")
        else:
            line += "; grounded rotation negligible (no support lever) → rotate in flight"
        L.append(line)
    L.append("    - aerial: a flip/twist needs ω ≥ 2π/t_air (see flight time above); "
             "the legs impart angular momentum L = I·ω at take-off. A ~9 kg quadruped "
             "(MIT Mini Cheetah) lands 360° backflips, so this regime is real.")
    L.append("")

    # ---- sources ----
    L.append("- Sources: Froude number & v=√(gL) — R. McN. Alexander / "
             "en.wikipedia.org/wiki/Froude_number; aerial rotation & backflip "
             "feasibility — MIT Mini Cheetah, Katz et al., ICRA 2019; Unitree Go2 "
             "top speed 2.5–5 m/s — unitree.com/go2. Torque source is the MJCF "
             "`actuatorfrcrange`; jump/rate force terms are statics estimates.")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--robots_dir", default="robots")
    ap.add_argument("--dynamics", action="store_true",
                    help="force the dynamics section even if the auto-gate fails")
    ap.add_argument("--no-dynamics", action="store_true",
                    help="force kinematics-only")
    args = ap.parse_args()

    spec = load_robot_spec(args.robot, robots_dir=args.robots_dir)
    from utils.scene_compose import compose_model
    m = compose_model(spec.mjcf, spec.scene_xml)   # robot + agnostic scene
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(m, d, kid)
    mujoco.mj_forward(m, d)

    # auto-decide the dynamics tier from what the model actually carries
    emit = rd.has_usable_dynamics(m, spec)
    if args.no_dynamics:
        emit = False
    elif args.dynamics:
        emit = True

    out = [f"# Physical summary — {spec.name}",
           f"(generated from {os.path.basename(spec.mjcf)}; do not edit by hand)", ""]
    out.append(kinematic_section(m, d, spec))
    if emit:
        out.append(dynamics_section(m, d, spec))
    else:
        out.append("## Dynamics\n\nThis model carries no actuator torque limits, so "
                    "velocity/acceleration ceilings are not derived — choose joint and "
                    "base velocities to look natural, within the kinematic limits above.\n")

    dest = Path(args.robots_dir) / args.robot / "summary.md"
    dest.write_text("\n".join(out))
    print(f"[summary] wrote {dest}  (dynamics={'on' if emit else 'off'})")

    # ONBOARDING SANITY: catch a home-pose IK singularity (a straight-leg import)
    # NOW, before the user wastes a run finding every skill's contacts float.
    folded = check_home_singularity(m, spec, kid)
    if folded:
        print("\n⚠️  HOME-POSE IK SINGULARITY — the <keyframe name=\"home\"> puts these "
              "limbs at/near FULL EXTENSION, so IK will FOLD them the wrong way when a "
              "skill reaches for a foothold (every contact on that limb will FLOAT):")
        for n, res, reach in folded:
            print(f"     - {n}: could not reach a target within {res*100:.0f} cm "
                  f"(more than half its {reach*100:.0f} cm reach) — the limb folded/stuck.")
        print("     FIX: edit the model's <keyframe name=\"home\"> to a BENT pose (flex the "
              "knees/elbows slightly) so IK has a bend direction, then re-run this tool.")
    else:
        print("[summary] home-pose IK conditioning OK (no limb folds from home).")

    # Machine-readable per-joint velocity/accel ceilings (rad/s, rad/s²) for the
    # joint_velocity_envelope unit-test check — the same numbers the summary text
    # cites, in a form RobotSpec reads directly. Only when the model carries a
    # torque source (else no ceilings exist and the check self-skips).
    # Machine-readable REACH ENVELOPE — always emitted (it is pure kinematics and
    # needs no torque source). For each end-effector it answers the ONE question the
    # planner must answer before it writes a contact schedule:
    #
    #   "how high may the BASE be for this limb to reach a contact resting on a
    #    surface at height H?"        base_z  <=  H + max_base_over_surface
    #
    # derived from  base_z + root_dz - contact_z <= reach  with contact_z =
    # H + ground_offset, where root_dz is the LIMB ROOT (shoulder/hip) offset, not the
    # end-effector's resting position.
    #
    # !! VALID FOR AN UPRIGHT BASE ONLY. root_dz is a BODY-frame offset, so pitching or
    # inverting the torso rotates the limb root and changes the answer completely — a
    # shoulder 0.48 m ABOVE an upright pelvis swings down to pelvis height at 90 deg of
    # pitch. That is exactly why `drop_below_base_upright` is reported alongside: for
    # the humanoid it is 0.83 m for a foot but only 0.05 m for a hand, so a hand plant
    # on the FLOOR is unreachable from any upright pose and REQUIRES the plan to pitch
    # or invert the torso. Without these numbers the planner writes the contact
    # schedule and the base trajectory independently and only discovers downstream that
    # they are incompatible, which is what makes every hand-planting skill (cartwheel,
    # handstand, vault, wall-kick) unbuildable as planned.
    from utils.reach import compute_limb_geometry as _clg
    _geom = _clg(m, d, spec.base_body, spec.ee_names, spec.ee_bodies)
    reach_env = {}
    for _i, _n in enumerate(spec.ee_names):
        _g = _geom.get(_n)
        if not _g or float(_g["reach"]) < 1e-6:
            continue
        _r = float(_g["reach"]); _dz = float(_g["root_offset"][2])
        _off = float(spec.ee_ground_offsets[_i])
        reach_env[_n] = {
            "reach": round(_r, 3),
            "root_dz": round(_dz, 3),               # limb ROOT (shoulder/hip) vs the base
            # full BODY-frame root offset, so a consumer can rotate it by the torso's
            # pitch instead of assuming upright (an inverted torso puts a shoulder
            # BELOW the pelvis, which is exactly how a cartwheel reaches the floor)
            "root_off": [round(float(x), 3) for x in _g["root_offset"]],
            "ground_offset": round(_off, 3),
            # base_z <= H + this, for an UPRIGHT base
            "max_base_over_surface": round(_off + _r - _dz, 3),
            # how far BELOW the base this limb's contact point reaches when upright —
            # the number that makes the constraint obvious at a glance
            "drop_below_base_upright": round(_r - _dz - _off, 3),
        }

    dyn_dest = Path(args.robots_dir) / args.robot / "dynamics.json"
    if emit:
        lim = rd.joint_accel_velocity_limits(m, d, spec)
        vel = {jn: round(r["qdot_peak"], 3) for jn, r in lim.items()
               if r.get("qdot_peak") and r.get("reliable")}
        acc = {jn: round(r["qddot_max"], 3) for jn, r in lim.items()
               if r.get("qddot_max") and r.get("reliable")}
        # ballistic jump estimate (leg-push apex RISE + airtime) — the physical scale
        # the flight_window plausibility check needs (an absolute z ceiling false-fails
        # a legal hop onto raised terrain and is vacuous on a tall robot).
        jr = rd.base_vertical_jump(m, d, spec)
        jump = ({"height": round(jr["jump_height"], 3), "flight_time": round(jr["flight_time"], 3)}
                if jr.get("jump_height") else {})
        dyn_dest.write_text(json.dumps({"vel": vel, "accel": acc, "jump": jump,
                                        "reach": reach_env}, indent=2))
        print(f"[summary] wrote {dyn_dest}  ({len(vel)} joint ceilings; "
              f"jump={jump.get('height', 'n/a')})")
    else:
        # Kinematics-only robot: still emit the reach envelope on its own, since it
        # is what the planner needs and it does not depend on the dynamics tier.
        dyn_dest.write_text(json.dumps({"reach": reach_env}, indent=2))
        print(f"[summary] wrote {dyn_dest}  (reach envelope only; no dynamics tier)")
        # (The file is always rewritten above, so turning the dynamics tier off no
        # longer leaves stale ceilings behind — the fresh write drops them.)


if __name__ == "__main__":
    main()
