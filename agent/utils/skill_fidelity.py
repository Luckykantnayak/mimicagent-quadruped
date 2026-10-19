"""Phase-specific intent checks, separate from the legacy legality scorer."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from utils.test_motion_quality import base_result

METRICS = {
    "displacement_x_m", "displacement_y_m", "displacement_z_m",
    "speed_x_m_s", "speed_y_m_s", "speed_z_m_s",
    "rotation_body_x_deg", "rotation_body_y_deg", "rotation_body_z_deg",
    "heading_rotation_deg", "heading_forward_fraction", "horizontal_excursion_m",
    "supported_heading_rotation_deg",
    "nose_up_min_deg", "nose_up_max_deg", "height_excursion_m",
    "support_fraction", "clearance_min_m", "lift_max_m", "airborne_fraction",
    "leg_motion_min_rad", "contact_alternations", "translation_step_max_m",
    "rotation_step_max_deg",
}
LEG_METRICS = {"support_fraction", "clearance_min_m", "lift_max_m", "leg_motion_min_rad", "contact_alternations", "supported_heading_rotation_deg"}


def validate_criteria(criteria):
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("success_criteria must be a nonempty list of measurable intent checks")
    names = set()
    for c in criteria:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or not c["name"] or c["name"] in names:
            raise ValueError("Each success criterion needs a unique nonempty name")
        names.add(c["name"])
        if c.get("metric") not in METRICS:
            raise ValueError(f"Unsupported success metric: {c.get('metric')}; choose from {sorted(METRICS)}")
        phase = c.get("phase")
        if not isinstance(phase, list) or len(phase) != 2 or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in phase) or not 0 <= phase[0] < phase[1] <= 1:
            raise ValueError("phase must be [start, end] within [0,1], with start < end")
        bounds = [c[k] for k in ("min", "max") if k in c]
        if not bounds or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in bounds):
            raise ValueError("Each criterion needs a finite numeric min and/or max")
        if c.get("min", -np.inf) > c.get("max", np.inf):
            raise ValueError("Criterion min exceeds max")
        if c["metric"] in LEG_METRICS:
            legs = c.get("legs")
            if not isinstance(legs, list) or not legs or len(set(legs)) != len(legs) or not all(v in ("FL", "FR", "RL", "RR") for v in legs):
                raise ValueError("Leg metric requires unique canonical legs from FL, FR, RL, RR")
        if c["metric"] == "heading_forward_fraction" and c.get("direction") not in (-1, 1):
            raise ValueError("heading_forward_fraction requires direction +1 or -1")
        if c["metric"] == "supported_heading_rotation_deg":
            raised = c.get("raised_legs")
            band = c.get("nose_up_range")
            if not isinstance(raised, list) or not raised or not all(v in ("FL", "FR", "RL", "RR") for v in raised) or set(raised) & set(c["legs"]):
                raise ValueError("supported heading requires disjoint supported legs and raised_legs")
            if not isinstance(band, list) or len(band) != 2 or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in band) or not -90 <= band[0] < band[1] <= 90:
                raise ValueError("supported heading requires nose_up_range=[min,max] within [-90,90]")
        if "min_support" in c and (type(c["min_support"]) is not int or not 1 <= c["min_support"] <= len(c.get("legs", []))):
            raise ValueError("min_support must be between 1 and the number of selected legs")


def achieved_kinematics(traj, repo_root):
    """Forward-kinematics only: never solve IK or edit the saved trajectory."""
    import mujoco
    root = Path(repo_root)
    sim_cfg = yaml.safe_load((root / "simulator/config.yaml").read_text())
    robot = sim_cfg["run"]["robot"]
    spec = yaml.safe_load((root / "robots" / robot / "robot.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(root / spec["scene_xml"]))
    data = mujoco.MjData(model)
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in spec["foot_sites"]]
    if min(sites) < 0:
        raise ValueError("Robot configuration references a missing foot site")
    feet = []
    for pos, quat, joints in zip(traj["root_pos"], traj["root_quat"], traj["joints"]):
        data.qpos[:] = np.r_[pos, quat, joints]
        mujoco.mj_fwdPosition(model, data)
        feet.append(data.site_xpos[sites].copy())
    joint_groups = {}
    for leg in spec["leg_order"]:
        indices = []
        for name in spec["joint_names"]:
            if name.startswith(leg + "_"):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise ValueError(f"Missing joint {name}")
                indices.append(int(model.jnt_qposadr[jid]) - 7)
        joint_groups[leg] = indices
    return np.asarray(feet), spec["leg_order"], spec.get("wheel_radius", 0.0), joint_groups


def complete_windows(phase, interval):
    """Ignore cropped boundary windows, but require at least one full window."""
    delta = np.diff(phase)
    step = np.median(delta[delta > 0]) if np.any(delta > 0) else 0
    if step <= 0 or step > .1:
        raise ValueError("Invalid or undersampled recorded phase")
    cycles = np.cumsum(np.r_[0, delta < -.5])
    a, b = interval
    for cycle in np.unique(cycles):
        indices = np.flatnonzero((cycles == cycle) & (phase >= a) & (phase < b))
        if len(indices) >= 2 and phase[indices[0]] <= a + 2 * step and phase[indices[-1]] >= b - 2 * step:
            yield indices


def evaluate_fidelity(traj, criteria, feet, leg_order, radius, joint_groups):
    validate_criteria(criteria)
    phase = np.asarray(traj.get("phase", []))
    if phase.shape != (len(traj["root_pos"]),) or not np.isfinite(phase).all() or np.any((phase < 0) | (phase >= 1)):
        raise ValueError("Fidelity requires finite recorded phase in [0,1)")
    pos = np.asarray(traj["root_pos"])
    joints = np.asarray(traj["joints"])
    rot = Rotation.from_quat(np.asarray(traj["root_quat"])[:, [1, 2, 3, 0]])
    matrices = rot.as_matrix()
    body_steps = np.rad2deg((rot[:-1].inv() * rot[1:]).as_rotvec())
    nose_up = np.rad2deg(np.arcsin(np.clip(matrices[:, 2, 0], -1, 1)))
    heading = np.unwrap(np.arctan2(-matrices[:, 0, 1], matrices[:, 1, 1]))
    height = np.asarray(feet)[:, :, 2] - radius
    contact = np.abs(height) <= .02
    results = []
    for c in criteria:
        r = base_result("intent:" + c["name"], True, "discrete")
        r["thresholds"] = c
        values = []
        for ids in complete_windows(phase, c["phase"]):
            start, end = ids[0], ids[-1]
            metric = c["metric"]
            legs = [leg_order.index(leg) for leg in c.get("legs", [])]
            if metric.startswith("displacement_"):
                value = pos[end, "xyz".index(metric.split("_")[1])] - pos[start, "xyz".index(metric.split("_")[1])]
            elif metric.startswith("speed_"):
                axis = "xyz".index(metric.split("_")[1])
                value = (pos[end, axis] - pos[start, axis]) * traj["fps"] / (end - start)
            elif metric.startswith("rotation_body_"):
                value = body_steps[start:end, "xyz".index(metric.split("_")[2])].sum()
            elif metric.startswith("heading_") or metric == "supported_heading_rotation_deg":
                if np.min(np.linalg.norm(matrices[ids, :2, 1], axis=1)) < .1:
                    raise ValueError("Heading undefined: lateral body axis is vertical; use body rotation metrics")
                steps = np.rad2deg(np.diff(heading[ids]))
                if metric == "supported_heading_rotation_deg":
                    raised = [leg_order.index(leg) for leg in c["raised_legs"]]
                    valid = ((contact[ids][:, legs].sum(axis=1) >= c.get("min_support", 1))
                             & np.all(height[ids][:, raised] >= .05, axis=1)
                             & (nose_up[ids] >= c["nose_up_range"][0])
                             & (nose_up[ids] <= c["nose_up_range"][1]))
                    value = steps[valid[:-1] & valid[1:]].sum()
                else:
                    value = steps.sum() if metric == "heading_rotation_deg" else np.mean(steps * c["direction"] * traj["fps"] > 1)
            elif metric == "horizontal_excursion_m":
                value = np.linalg.norm(np.ptp(pos[ids, :2], axis=0))
            elif metric == "nose_up_min_deg":
                value = nose_up[ids].min()
            elif metric == "nose_up_max_deg":
                value = nose_up[ids].max()
            elif metric == "height_excursion_m":
                value = np.ptp(pos[ids, 2])
            elif metric == "support_fraction":
                value = np.mean(contact[ids][:, legs].sum(axis=1) >= c.get("min_support", 1))
            elif metric == "clearance_min_m":
                value = height[ids][:, legs].min()
            elif metric == "lift_max_m":
                value = height[ids][:, legs].max()
            elif metric == "airborne_fraction":
                value = np.mean(np.all(height[ids] > .02, axis=1))
            elif metric == "leg_motion_min_rad":
                value = min(np.ptp(joints[ids][:, joint_groups[leg]], axis=0).max() for leg in c["legs"])
            elif metric == "contact_alternations":
                value = min(np.count_nonzero(np.diff(contact[ids, leg].astype(int))) for leg in legs)
            elif metric == "translation_step_max_m":
                # Include boundary steps, especially the phase=1 -> 0 seam.
                value = np.linalg.norm(np.diff(pos[max(0, start-1):min(len(pos), end+2)], axis=0), axis=1).max()
            elif metric == "rotation_step_max_deg":
                value = np.linalg.norm(body_steps[max(0, start-1):min(len(body_steps), end+1)], axis=1).max()
            values.append(float(value))
        r["metrics"] = {"complete_window_values": values}
        r["passed"] = bool(values) and all(np.isfinite(v) and c.get("min", -np.inf) <= v <= c.get("max", np.inf) for v in values)
        r["reward"] = 0.0 if r["passed"] else -1.0
        r["feedback"] = (f"{c['metric']} in phase {c['phase']}: {values}; required "
                         f"min={c.get('min')}, max={c.get('max')}. "
                         + ("" if values else "No complete phase window recorded."))
        if not r["passed"]:
            r["violations"] = [r["feedback"]]
        results.append(r)
    return results


def append_fidelity(report, traj, criteria, repo_root):
    # Old plans preserve their prior legality scores, explicitly unverified for intent.
    report["summary"]["fidelity_checked"] = criteria is not None
    if criteria is None:
        return report
    try:
        feet, legs, radius, groups = achieved_kinematics(traj, repo_root)
        checks = evaluate_fidelity(traj, criteria, feet, legs, radius, groups)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        check = base_result("intent:invalid_evidence", True, "discrete")
        check.update(passed=False, reward=-1.0, violations=[str(exc)], feedback=str(exc))
        checks = [check]
    report["tests"].extend(checks)
    summary = report["summary"]
    failed = sum(not c["passed"] for c in checks)
    summary["passed_all"] = bool(summary["passed_all"] and not failed)
    summary["passed_all_discrete"] = bool(summary["passed_all_discrete"] and not failed)
    summary["num_tests"] += len(checks)
    summary["num_failed"] += failed
    summary["total_reward"] += sum(c["reward"] for c in checks)
    return report
