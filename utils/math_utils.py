import numpy as np
from scipy.spatial.transform import Rotation as R


# =====================================================
#           MATH UTILITIES
# =====================================================

def quat_to_rotation_matrix(quat):
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x,y,z,w]
    return rot.as_matrix()


def rotation_matrix_to_quat(mat):
    """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
    rot = R.from_matrix(mat)
    q = rot.as_quat()  # [x,y,z,w]
    return np.array([q[3], q[0], q[1], q[2]])  # [w,x,y,z]


def integrate_pose_world_frame(pos, quat, vel_world, omega_world, dt):
    """
    Integrate pose given world-frame velocities.

    Args:
        pos: current position [x,y,z]
        quat: current orientation [w,x,y,z]
        vel_world: linear velocity in world frame [vx, vy, vz]
        omega_world: angular velocity in world frame [wx, wy, wz] (rad/s)
        dt: timestep

    Returns:
        new_pos: updated position
        new_quat: updated orientation
    """
    # Linear integration
    new_pos = pos + vel_world * dt

    # Angular integration using rotation matrix
    R_current = quat_to_rotation_matrix(quat)

    # Create rotation matrix from angular velocity
    angle = np.linalg.norm(omega_world) * dt
    if angle > 1e-8:
        axis = omega_world / np.linalg.norm(omega_world)
        rot_delta = R.from_rotvec(axis * angle)
        R_new = rot_delta.as_matrix() @ R_current
    else:
        R_new = R_current

    new_quat = rotation_matrix_to_quat(R_new)

    return new_pos, new_quat


def world_to_body_velocity(vel_world, quat):
    """
    Convert world frame velocity to body frame.

    Args:
        vel_world: velocity in world frame [vx, vy, vz]
        quat: body orientation [w,x,y,z]

    Returns:
        vel_body: velocity in body frame
    """
    R_mat = quat_to_rotation_matrix(quat)
    return R_mat.T @ vel_world


def body_to_world_position(pos_body, root_pos, root_quat):
    """
    Convert body frame position to world frame.

    Args:
        pos_body: position in body frame [x,y,z]
        root_pos: root position in world frame
        root_quat: root orientation [w,x,y,z]

    Returns:
        pos_world: position in world frame
    """
    R_mat = quat_to_rotation_matrix(root_quat)
    return root_pos + R_mat @ pos_body


def quat_to_euler(q):
    """
    Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw)
    """
    w, x, y, z = q
    # Roll
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    # Pitch
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    # Yaw
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion [w, x, y, z]
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def quat_to_exp_map(quat):
    """
    Convert a quaternion to exponential map (axis-angle) representation.

    Args:
        quat: array-like, shape (4,) quaternion [x, y, z, w]

    Returns:
        exp_map: np.array shape (3,) exponential map vector
    """
    x, y, z, w = quat
    # Normalize quaternion
    norm = np.linalg.norm([x, y, z, w])
    if norm > 0:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    else:
        return np.zeros(3)

    # Compute angle
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))

    # Avoid division by zero
    s = np.sqrt(1 - w * w)
    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])  # Arbitrary axis
    else:
        axis = np.array([x, y, z]) / s

    exp_map = axis * angle
    return exp_map


def quat_multiply(q1, q2):
    """
    Quaternion multiplication.

    Args:
        q1, q2: quaternions in [w, x, y, z] format

    Returns:
        q = q1 ⊗ q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


