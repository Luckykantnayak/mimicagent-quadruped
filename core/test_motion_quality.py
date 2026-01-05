import numpy as np
from utils.math_utils import (
    quat_relative,
    quat_angle
)

# =====================================================
#          LOAD TRAJECTORY
# =====================================================

def load_traj(path):
    traj = np.load(path, allow_pickle=True)
    return traj

# =====================================================
# 1.1 END-CONDITION CONSISTENCY (CYCLE CLOSURE) - SKILL DEPENDENT
# =====================================================

def test_configuration_closure(
    traj,
    joint_tol=0.15,        # radians
    orientation_tol=np.deg2rad(10)
):
    q0 = traj["joints"][0]
    qT = traj["joints"][-1]

    joint_delta = np.linalg.norm(qT - q0) / len(q0)

    assert joint_delta < joint_tol, \
        f"Joint configuration did not close: Δq={joint_delta:.3f} rad"

    q0_base = traj["root_quat"][0]
    qT_base = traj["root_quat"][-1]

    q_rel = quat_relative(q0_base, qT_base)
    delta_angle = quat_angle(q_rel)

    assert delta_angle < orientation_tol, \
        f"Base orientation did not close: Δθ={np.rad2deg(delta_angle):.2f}°"


# =====================================================
# 1.2 RELATIVE-FOOT CLOSURE (CYCLE CLOSURE) - SKILL DEPENDENT
# =====================================================

def test_relative_foot_closure(traj, tol=0.05):
    feet0 = traj["foot_positions"][0]
    feetT = traj["foot_positions"][-1]

    base0 = traj["root_pos"][0]
    baseT = traj["root_pos"][-1]

    for leg in feet0.keys():
        rel0 = feet0[leg] - base0
        relT = feetT[leg] - baseT
        delta = np.linalg.norm(relT - rel0)

        assert delta < tol, \
            f"{leg} relative foot pose did not close: {delta:.3f} m"


# =====================================================
# 1.3 BASE HEIGHT ENVELOPE TEST -- UNIVERSAL
# =====================================================

def test_base_height_envelope(traj, z_min=0.10, z_max=0.68):
    z = traj["root_pos"][:, 2]

    z_lo = z.min()
    z_hi = z.max()

    assert z_lo >= z_min, \
        f"Base collapsed: z_min={z_lo:.3f} m"

    assert z_hi <= z_max, \
        f"Base too high: z_max={z_hi:.3f} m"

# =====================================================
# 1.4 ORIENTATION CONTINUITY TEST - UNIVERSAL
# =====================================================

def test_orientation_continuity(traj, max_step_angle=np.deg2rad(10)):
    quats = traj["root_quat"]

    for i in range(len(quats) - 1):
        q_rel = quat_relative(quats[i], quats[i + 1])
        angle = quat_angle(q_rel)

        assert angle < max_step_angle, \
            f"Orientation jump at step {i}: {np.rad2deg(angle):.2f} deg"

# =====================================================
# 2.2 FOOT VELOCITY SMOOTHNESS TEST - UNIVERSAL
# =====================================================

def test_foot_velocity_smoothness(
    traj,
    dt,
    vmax_percentile=99.9,
    abs_cap=10.0
):
    feet = traj["foot_positions"]
    leg_names = feet[0].keys()

    for leg in leg_names:
        positions = np.array([f[leg] for f in feet])
        vel = np.linalg.norm(np.diff(positions, axis=0) / dt, axis=1)

        v_p = np.percentile(vel, vmax_percentile)
        v_max = vel.max()

        assert v_p < abs_cap, \
            f"{leg} extreme foot velocity: p{vmax_percentile}={v_p:.2f} m/s"

        # Optional hard teleport check
        assert v_max < 2 * abs_cap, \
            f"{leg} foot teleport detected: vmax={v_max:.2f} m/s"


# =====================================================
# 2.3 GROUND PENETRATION TEST (SOFT) - UNIVERSAL
# =====================================================

def test_ground_penetration(traj, ground_z=0.0, tol=0.01):
    feet = traj["foot_positions"]

    for t, foot_dict in enumerate(feet):
        for leg, pos in foot_dict.items():
            assert pos[2] >= ground_z - tol, \
                f"{leg} penetrated ground at t={t}, z={pos[2]:.3f}"

# =====================================================
# 3.1 JOINT LIMIT VIOLATION TEST - UNIVERSAL
# =====================================================

def test_joint_limits(traj, joint_limits):
    """
    joint_limits: dict {joint_name: (min, max)}
    """
    q = traj["joints"]

    for i, (jmin, jmax) in enumerate(joint_limits.values()):
        q_i = q[:, i]
        assert q_i.min() >= jmin, f"Joint {i} below limit"
        assert q_i.max() <= jmax, f"Joint {i} above limit"

# =====================================================
# 3.2 JOINT VELOCITY SPIKE TEST - UNIVERSAL
# =====================================================

def test_joint_velocity_outliers(
    traj,
    dt,
    z_thresh=5.0,
    min_frac=0.005
):
    """
    Detect abnormal joint velocity spikes via z-score.

    Args:
        z_thresh: z-score threshold (5σ is conservative)
        min_frac: fraction of samples allowed to exceed threshold
    """
    q = traj["joints"]
    qd = np.diff(q, axis=0) / dt  # (T-1, J)

    mean = np.mean(qd, axis=0)
    std = np.std(qd, axis=0) + 1e-6

    z = np.abs((qd - mean) / std)

    outlier_frac = np.mean(z > z_thresh)

    assert outlier_frac < min_frac, \
        f"Joint velocity spikes detected: {outlier_frac*100:.2f}% samples > {z_thresh}σ"


# =====================================================
# 3.3 JOINT ACCELERATION SMOOTHNESS TEST - UNIVERSAL
# =====================================================

def test_joint_acceleration_outliers(
    traj,
    dt,
    z_thresh=6.0,
    max_percentile=99.9
):
    q = traj["joints"]
    qd = np.diff(q, axis=0) / dt
    qdd = np.diff(qd, axis=0) / dt

    mean = np.mean(qdd, axis=0)
    std = np.std(qdd, axis=0) + 1e-6
    z = np.abs((qdd - mean) / std)

    z_max = np.max(z)
    p99 = np.percentile(np.abs(qdd), max_percentile)

    assert z_max < z_thresh, \
        f"Joint acceleration spike: max z={z_max:.1f}"

    assert p99 < 5 * np.mean(np.abs(qdd)), \
        f"Acceleration distribution heavy-tailed (p99={p99:.1f})"


# =====================================================
# 3.4 JOINT JERK SMOOTHNESS TEST - UNIVERSAL
# =====================================================

def test_joint_jerk_smoothness(
    traj,
    dt,
    z_thresh=6.0
):
    q = traj["joints"]
    qd = np.diff(q, axis=0) / dt
    qdd = np.diff(qd, axis=0) / dt
    qddd = np.diff(qdd, axis=0) / dt

    std = np.std(qddd, axis=0) + 1e-6
    z = np.abs(qddd / std)

    assert np.max(z) < z_thresh, \
        f"High jerk detected (z>{z_thresh})"


# =====================================================
# 3.5 ALL-FEET AIRTIME TEST (CONTACT-BASED) - SKILL DEPENDENT
# =====================================================

def test_all_feet_airtime(
    traj,
    dt,
    max_airtime=0.24
):
    """
    Ensures that all four feet are not simultaneously off the ground
    for longer than max_airtime seconds, using stored contact data.
    """

    contacts = traj["foot_contacts"]

    current_airtime = 0.0
    max_observed_airtime = 0.0

    for t, contact_dict in enumerate(contacts):
        all_off_ground = True

        for leg, in_contact in contact_dict.items():
            if in_contact:
                all_off_ground = False
                break

        if all_off_ground:
            current_airtime += dt
            max_observed_airtime = max(max_observed_airtime, current_airtime)
        else:
            current_airtime = 0.0

    assert max_observed_airtime <= max_airtime, \
        (
            f"Flight phase detected: all feet airborne for "
            f"{max_observed_airtime:.3f}s "
            f"(limit {max_airtime:.3f}s)"
        )

