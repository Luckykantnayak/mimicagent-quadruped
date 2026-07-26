import os
import re

import numpy as np
import json
from typing import Dict, Any


# =====================================================
#                MATH UTILS
# =====================================================

def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_relative(q1, q2):
    return quat_multiply(q2, quat_conjugate(q1))


def quat_angle(q):
    w = np.clip(q[0], -1.0, 1.0)
    return 2.0 * np.arccos(abs(w))


# =====================================================
#          TRAJECTORY LOADING
# =====================================================

def load_traj(path: str) -> Dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


# =====================================================
#          RESULT TEMPLATE
# =====================================================

def base_result(name: str, skill_dependent: bool, category: str):
    return dict(
        name=name,
        category=category,  # "discrete" | "smoothness"
        skill_dependent=skill_dependent,
        passed=True,
        skipped=False,
        reward=0.0,
        metrics={},
        thresholds={},
        violations=[],
        feedback=""
    )


# =====================================================
#          DISCRETE / HARD CONSTRAINTS
# =====================================================

def eval_configuration_closure(traj, joint_tol=0.15, ori_tol=np.deg2rad(10)):
    r = base_result("configuration_closure", True, "discrete")

    q_full = traj["joints"]

    # Indices of passive wheel joints in traj["joints"]
    passive_joints = {3, 7, 11, 15}

    # Keep only actuated joints, preserving order
    actuated_indices = [i for i in range(q_full.shape[1]) if i not in passive_joints]
    q = q_full[:, actuated_indices]

    q0, qT = q[0], q[-1]
    joint_delta = np.linalg.norm(qT - q0) / len(q0)

    q_rel = quat_relative(traj["root_quat"][0], traj["root_quat"][-1])
    ori_delta = quat_angle(q_rel)

    r["metrics"] = dict(
        joint_delta=float(joint_delta),
        orientation_delta_deg=float(np.rad2deg(ori_delta))
    )
    r["thresholds"] = dict(
        joint_tol=joint_tol,
        orientation_tol_deg=float(np.rad2deg(ori_tol))
    )

    joint_excess = max(0.0, joint_delta - joint_tol)
    ori_excess = max(0.0, ori_delta - ori_tol)

    r["reward"] = -5.0 * joint_excess - 3.0 * ori_excess
    r["passed"] = (joint_excess == 0.0 and ori_excess == 0.0)

    if not r["passed"]:
        r["violations"].append("Cyclic closure violated.")

    r["feedback"] = (
        "Cyclic closure satisfied within tolerance."
        if r["passed"]
        else (
            f"Joint drift {joint_delta:.3f} (tol {joint_tol}), "
            f"orientation drift {np.rad2deg(ori_delta):.2f}° "
            f"(tol {np.rad2deg(ori_tol):.2f}°)."
        )
    )

    return r


def eval_relative_foot_closure(traj, tol=0.05):
    r = base_result("relative_foot_closure", True, "discrete")

    feet0 = traj["foot_positions"][0]    # (4, 3)
    feetT = traj["foot_positions"][-1]   # (4, 3)
    base0 = traj["root_pos"][0]
    baseT = traj["root_pos"][-1]

    excess = {}
    for leg_idx in range(feet0.shape[0]):  # 0=FL, 1=FR, 2=RL, 3=RR
        d = np.linalg.norm(
            (feetT[leg_idx] - baseT) -
            (feet0[leg_idx] - base0)
        )
        excess[leg_idx] = max(0.0, d - tol)

    r["metrics"]["per_leg_excess"] = excess
    r["thresholds"]["tol"] = tol

    r["reward"] = -2.0 * sum(v * v for v in excess.values())
    r["passed"] = all(v == 0.0 for v in excess.values())

    if not r["passed"]:
        r["violations"] = [
            leg for leg, v in excess.items() if v > 0
        ]

    r["feedback"] = (
        "Relative foot closure satisfied."
        if r["passed"]
        else "Relative foot drift detected."
    )
    return r


def eval_base_height(traj, z_min=0.10, z_max=0.68):
    r = base_result("base_height_envelope", False, "discrete")

    z = traj["root_pos"][:, 2]
    z_lo, z_hi = z.min(), z.max()

    r["metrics"] = dict(z_min=float(z_lo), z_max=float(z_hi))
    r["thresholds"] = dict(z_min=z_min, z_max=z_max)

    if z_lo < z_min or z_hi > z_max:
        r["passed"] = False
        r["reward"] = -1.0
        r["violations"].append("Base height envelope violated.")

    r["feedback"] = (
        "Base height within limits."
        if r["passed"]
        else f"Base height unsafe. The base height varied from {z_lo} to {z_hi} while the allowed range is {z_min} to {z_max}."
    )
    return r


def eval_ground_penetration(traj, ground_z=0.0, tol=0.02):
    r = base_result("ground_penetration", False, "discrete")

    first_t = None
    first_leg = None
    deepest = 0.0
    wheel_radius = 0.10

    # Per-leg stats (0=FL, 1=FR, 2=RL, 3=RR)
    per_leg_violation_count = {0: 0, 1: 0, 2: 0, 3: 0}
    per_leg_max_penetration = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    violations = []

    # foot_positions: (T, 4, 3)
    for t, feet in enumerate(traj["foot_positions"]):
        for leg_idx in range(feet.shape[0]):
            z = feet[leg_idx, 2] + wheel_radius
            penetration = ground_z - z

            # Track per-leg max penetration (even if below tol)
            per_leg_max_penetration[leg_idx] = max(
                per_leg_max_penetration[leg_idx], penetration
            )
            deepest = max(deepest, penetration)

            if penetration > tol:
                per_leg_violation_count[leg_idx] += 1

                # Record first violation only
                if first_t is None:
                    first_t = t
                    first_leg = leg_idx
                    violations.append(
                        dict(
                            t=int(t),
                            leg=int(leg_idx),
                            penetration=float(penetration),
                        )
                    )

    total_violations = sum(per_leg_violation_count.values())

    r["metrics"] = dict(
        num_violations=int(total_violations),
        first_violation_t=first_t,
        first_violation_leg=first_leg,
        max_penetration=float(deepest),
        per_leg_violation_count=per_leg_violation_count,
        per_leg_max_penetration=per_leg_max_penetration,
    )

    r["violations"] = violations
    r["passed"] = total_violations == 0
    r["reward"] = 0.0 if r["passed"] else -1.0

    if r["passed"]:
        r["feedback"] = "No ground penetration detected throughout the trajectory."
    else:
        r["feedback"] = (
            f"Ground penetration detected. "
            f"Total violations: {total_violations}. "
            f"First violation at timestep {first_t} on leg {first_leg}. "
            f"Maximum penetration depth: {deepest:.3f} m."
        )

    return r



def eval_joint_limits(traj, joint_limits):
    r = base_result("joint_limits", False, "discrete")

    q_full = traj["joints"]

    # Indices of passive wheel joints in traj["joints"]
    passive_joints = {3, 7, 11, 15}

    # Keep only actuated joints, preserving order
    actuated_indices = [i for i in range(q_full.shape[1]) if i not in passive_joints]
    q = q_full[:, actuated_indices]

    violated = []

    for i, (jmin, jmax) in enumerate(joint_limits.values()):
        if q[:, i].min() < jmin or q[:, i].max() > jmax:
            violated.append(i)

    r["metrics"]["violated_joints"] = violated
    r["passed"] = len(violated) == 0
    r["reward"] = 0.0 if r["passed"] else -1.0
    r["feedback"] = (
        "Joint limits respected."
        if r["passed"]
        else "Joint limits violated."
    )
    return r



def eval_all_feet_airtime(traj, dt, max_airtime=0.5, foot_height_thresh=0.04):
    """
    Evaluates if all feet are airborne for too long.

    Contacts are dynamically computed from foot_positions and a height threshold.
    """
    r = base_result("all_feet_airtime", True, "discrete")

    airtime = 0.0
    max_air = 0.0

    wheel_radius = 0.10

    # foot_positions: (T, 4, 3)
    for feet in traj["foot_positions"]:
        # Compute contacts: foot is in contact if z <= threshold
        contacts = feet[:, 2] + wheel_radius <= foot_height_thresh  # shape (4,)

        if np.all(~contacts):  # all feet airborne
            airtime += dt
            max_air = max(max_air, airtime)
            if airtime > max_airtime:
                r["passed"] = False
                r["violations"].append(
                    f"All feet airborne for {max_air:.3f}s"
                )
                r["reward"] = -2.0
                r["feedback"] = "Flight phase detected."
                return r
        else:
            airtime = 0.0

    r["metrics"]["max_airtime"] = float(max_air)
    r["thresholds"]["max_airtime"] = max_airtime
    r["feedback"] = "No flight phase."
    return r



# =====================================================
#          SMOOTHNESS METRICS
# =====================================================

def _z_excess(x, thresh):
    z = np.abs((x - x.mean(0)) / (x.std(0) + 1e-6))
    return np.maximum(0.0, z - thresh)


def eval_joint_velocity(traj, dt, z_thresh=7.0, weight=0.1):
    r = base_result("joint_velocity_smoothness", False, "smoothness")

    # qd: (T-1, num_joints)

    q_full = traj["joints"]

    # Indices of passive wheel joints in traj["joints"]
    passive_joints = {3, 7, 11, 15}

    # Keep only actuated joints, preserving order
    actuated_indices = [i for i in range(q_full.shape[1]) if i not in passive_joints]
    q = q_full[:, actuated_indices]

    qd = np.diff(q, axis=0) / dt
    excess = _z_excess(qd, z_thresh)  # same shape

    # ---- per-leg aggregation ----
    # assumes joints are ordered per leg
    # shape: (T-1, num_legs)
    excess_per_leg = excess.reshape(excess.shape[0], 4, -1).mean(axis=2)

    # ---- robust statistics ----
    p99 = np.percentile(excess_per_leg, 99, axis=0)   # per leg
    mean_excess = np.mean(excess_per_leg)

    penalty = weight * mean_excess**2

    r["reward"] = -penalty
    r["metrics"]["mean_excess"] = float(mean_excess)
    r["metrics"]["p99_excess_per_leg"] = p99.tolist()
    r["thresholds"]["z_thresh"] = z_thresh

    # Pass if no leg is extremely spiky
    r["passed"] = np.max(p99) < 5.0
    r["feedback"] = (
        "Velocity smooth (robust)." if r["passed"] else "Velocity spikes (robust)."
    )

    return r



def eval_joint_acceleration(traj, dt, z_thresh=8.0, weight=0.2):
    r = base_result("joint_acceleration_smoothness", False, "smoothness")

    # qdd: (T-2, num_joints)

    q_full = traj["joints"]

    # Indices of passive wheel joints in traj["joints"]
    passive_joints = {3, 7, 11, 15}

    # Keep only actuated joints, preserving order
    actuated_indices = [i for i in range(q_full.shape[1]) if i not in passive_joints]
    q = q_full[:, actuated_indices]

    qd = np.diff(q, axis=0) / dt
    qdd = np.diff(qd, axis=0) / dt
    excess = _z_excess(qdd, z_thresh)

    # ---- per-leg aggregation ----
    excess_per_leg = excess.reshape(excess.shape[0], 4, -1).mean(axis=2)

    # ---- robust statistics ----
    p99 = np.percentile(excess_per_leg, 99, axis=0)
    mean_excess = np.mean(excess_per_leg)

    penalty = weight * mean_excess**2

    r["reward"] = -penalty
    r["metrics"]["mean_excess"] = float(mean_excess)
    r["metrics"]["p99_excess_per_leg"] = p99.tolist()
    r["thresholds"]["z_thresh"] = z_thresh

    # More permissive for impulsive gaits
    r["passed"] = np.max(p99) < 6.0
    r["feedback"] = (
        "Acceleration smooth (robust)." if r["passed"] else "Acceleration spikes (robust)."
    )

    return r



def eval_joint_jerk_energy_normalized(
    traj,
    dt,
    z_thresh=5.5,
    weight=0.3
):
    r = base_result(
        "joint_jerk_energy_normalized",
        False,
        "smoothness"
    )

    q_full = traj["joints"]

    # Indices of passive wheel joints in traj["joints"]
    passive_joints = {3, 7, 11, 15}

    # Keep only actuated joints, preserving order
    actuated_indices = [i for i in range(q_full.shape[1]) if i not in passive_joints]
    q = q_full[:, actuated_indices]

    qd = np.diff(q, axis=0) / dt
    qdd = np.diff(qd, axis=0) / dt
    qddd = np.diff(qdd, axis=0) / dt

    # RMS velocity = motion energy proxy
    vel_rms = np.sqrt(np.mean(qd**2)) + 1e-6

    jerk_norm = qddd / vel_rms
    excess = _z_excess(jerk_norm, z_thresh)

    penalty = weight * np.mean(excess**2)

    r["reward"] = -penalty
    r["metrics"]["mean_excess"] = float(np.mean(excess))
    r["metrics"]["vel_rms"] = float(vel_rms)
    r["passed"] = np.max(excess) < 4.0
    r["feedback"] = (
        "Energy-normalized jerk smooth."
        if r["passed"]
        else "High energy-normalized jerk."
    )

    return r


# =====================================================
#          AGGREGATION LOGIC
# =====================================================

def aggregate_discrete(results):
    discrete = [
        r for r in results
        if r["category"] == "discrete" and not r["skipped"]
    ]

    return dict(
        passed=all(r["passed"] for r in discrete),
        failed_tests=[r["name"] for r in discrete if not r["passed"]]
    )


def aggregate_smoothness(results, percentile=90, max_allowed_mean=1.5):
    smooth = [
        r for r in results
        if r["category"] == "smoothness" and not r["skipped"]
    ]

    if len(smooth) == 0:
        return dict(passed=True)

    mean_excess = np.array(
        [r["metrics"].get("mean_excess", 0.0) for r in smooth]
    )

    perc = np.percentile(mean_excess, percentile)
    worst = float(mean_excess.max())

    return dict(
        passed=(worst <= max(perc, max_allowed_mean)),
        worst_mean_excess=worst,
        percentile_threshold=float(perc)
    )


# =====================================================
#          MASTER EVALUATOR
# =====================================================

def evaluate_trajectory(
    traj,
    joint_limits,
    dt,
    skill_test_config: Dict[str, bool]
) -> Dict[str, Any]:

    evaluators = [
        # eval_configuration_closure,
        # eval_relative_foot_closure,
        eval_base_height,
        eval_ground_penetration,
        lambda t: eval_joint_limits(t, joint_limits),
        # lambda t: eval_joint_velocity(t, dt),
        # lambda t: eval_joint_acceleration(t, dt),
        # lambda t: eval_joint_jerk_energy_normalized(t, dt),
        lambda t: eval_all_feet_airtime(t, dt)
    ]

    results = []
    total_reward = 0.0

    for fn in evaluators:
        r = fn(traj)

        if r["skill_dependent"] and not skill_test_config.get(r["name"], False):
            r["skipped"] = True
            r["feedback"] = "Skipped (skill disabled)."
            results.append(r)
            continue

        total_reward += r["reward"]
        results.append(r)

    discrete_summary = aggregate_discrete(results)
    smoothness_summary = aggregate_smoothness(results)

    return dict(
        summary=dict(
            passed_all=(
                discrete_summary["passed"]
                and smoothness_summary["passed"]
            ),
            passed_all_discrete=discrete_summary["passed"],
            passed_smoothness=smoothness_summary["passed"],
            smoothness_stats=smoothness_summary,
            total_reward=float(total_reward),
            num_tests=len(results),
            num_failed=len(discrete_summary["failed_tests"])
        ),
        tests=results
    )


# =====================================================
#          SAVE REPORT
# =====================================================

def save_report(report: Dict[str, Any], out_dir: str, skill_name: str) -> str:
    return save_versioned_json(
        data=report,
        out_dir=out_dir,
        base_name=skill_name,
    )


def save_versioned_json(
    data: dict,
    out_dir: str,
    base_name: str,
    ext: str = ".json",
    pad: int = 4,
) -> str:
    """
    Saves data as base_name_XXXX.json in out_dir, where XXXX is the next version.
    Handles numpy types (bool, int, float) automatically.
    """
    os.makedirs(out_dir, exist_ok=True)

    pattern = re.compile(rf"{re.escape(base_name)}_(\d+){re.escape(ext)}$")
    versions = []

    for fname in os.listdir(out_dir):
        match = pattern.match(fname)
        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions) + 1 if versions else 1
    fname = f"{base_name}_{next_version:0{pad}d}{ext}"
    file_path = os.path.join(out_dir, fname)

    # Convert numpy types to native Python
    def convert(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Cannot serialize object of type {type(o)}")

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=convert)

    return file_path
