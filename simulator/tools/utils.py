import numpy as np
import mujoco
# ----------------------------
# Utility: body frame transform
# ----------------------------

from scipy.spatial.transform import Rotation as R

def euler_to_quat(roll=0, pitch=0, yaw=0):
    """Convert Euler angles to quaternion (w, x, y, z)"""
    r = R.from_euler('xyz', [roll, pitch, yaw])
    q = r.as_quat()  # (x,y,z,w)
    return np.array([q[3], q[0], q[1], q[2]])  # convert to w,x,y,z


def world_to_body(model, data, point_world):
    """Convert a 3D point from world frame to body frame."""
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    R_wb = data.xmat[trunk_id].reshape(3, 3)  # rotation from body to world
    p_wb = data.xpos[trunk_id]
    return R_wb.T @ (point_world - p_wb)  # inverse rotation + translation removal

def compute_phase(t, freq):
    """Phase ∈ [0, 1] for gait cycle."""
    return (freq * t) % 1.0


def quat_mul(q, p):
    w, x, y, z = q
    wp, xp, yp, zp = p
    return np.array([
        w*wp - x*xp - y*yp - z*zp,
        w*xp + x*wp + y*zp - z*yp,
        w*yp - x*zp + y*wp + z*xp,
        w*zp + x*yp - y*xp + z*wp
    ])

def integrate_yaw(q0, yaw_rate, dt):
    dpsi = yaw_rate * dt
    q_delta = np.array([
        np.cos(dpsi / 2),
        0.0,
        0.0,
        np.sin(dpsi / 2)
    ])
    q1 = quat_mul(q0, q_delta)
    return q1 / np.linalg.norm(q1)

def quat_to_euler(q):
    w, x, y, z = q

    # Roll (x-axis)
    roll = np.arctan2(
        2*(w*x + y*z),
        1 - 2*(x*x + y*y)
    )

    # Pitch (y-axis)
    sinp = 2*(w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis)
    yaw = np.arctan2(
        2*(w*z + x*y),
        1 - 2*(y*y + z*z)
    )

    return roll, pitch, yaw


def quat_pos(q):
    """
    Ensure quaternion has positive scalar part (q[...,3]).
    q: [..., 4]
    returns q: [...,4]
    """
    q = np.asarray(q)
    sign = np.where(q[..., 3:] < 0, -1.0, 1.0)
    return q * sign


def quat_to_axis_angle(q, eps=1e-5):
    """
    Convert quaternion -> (axis, angle)
    q: [..., 4]  quaternion in [x,y,z,w] or [w,x,y,z]?
    
    YOUR ORIGINAL TORCH VERSION USES:
        q[..., qx:qw] = x,y,z
        q[..., qw]    = w
    So expected input layout = [x, y, z, w]   ← MUJOCO FORMAT
    """
    q = quat_pos(q)
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]

    # vector part norm
    v = np.stack([x, y, z], axis=-1)
    length = np.linalg.norm(v, axis=-1)

    # angle
    angle = 2.0 * np.arctan2(length, w)

    # axis (safe divide)
    axis = np.zeros_like(v)
    mask = length > eps
    axis[mask] = v[mask] / length[mask][..., None]

    # default axis if angle is tiny
    default_axis = np.zeros_like(v)
    default_axis[..., 2] = 1.0
    axis[~mask] = default_axis[~mask]

    # zero angle for tiny rotations
    angle = np.where(mask, angle, np.zeros_like(angle))

    return axis, angle


def axis_angle_to_exp_map(axis, angle):
    """
    axis: [...,3]
    angle: [...]
    """
    angle = np.asarray(angle)[..., None]
    return axis * angle


def quat_to_exp_map(q):
    """
    Final mapping: quaternion -> exponential map
    q: [..., 4]  in MUJOCO format [x,y,z,w]
    """
    axis, angle = quat_to_axis_angle(q)
    return axis_angle_to_exp_map(axis, angle)


def rotate_about_z(p, center_xy, dtheta):
    c, s = np.cos(dtheta), np.sin(dtheta)
    R = np.array([[c, -s], [s, c]])
    v = p[:2] - center_xy
    p[:2] = center_xy + R @ v
