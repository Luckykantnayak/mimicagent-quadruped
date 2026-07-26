from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_WOBBLE_PLATE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Wobble Plate Spin: Continuous 360° yaw rotation with synchronized pitch-roll wobbling.
    
    - Base rotates continuously in yaw (constant angular velocity)
    - Pitch and roll oscillate in coordinated quadrant pattern
    - All four feet maintain continuous ground contact
    - Feet adjust dynamically in body frame to track wobbling base
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.6
        
        # Base foot positions (body frame reference)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Store nominal world-frame foot positions (assuming initial flat orientation)
        self.nominal_feet_world = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        for leg in self.nominal_feet_world:
            self.nominal_feet_world[leg][2] = 0.0  # All feet at ground level
        
        # Wobble parameters (reduced for workspace safety)
        self.pitch_amp = np.deg2rad(10)
        self.roll_amp = np.deg2rad(10)
        self.yaw_rate = 2 * np.pi * self.freq
        
        # Foot adjustment amplitude (reduced)
        self.foot_adjust_x = 0.018
        self.foot_adjust_y = 0.012
        self.yaw_adjust = 0.005
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base angular velocities: constant yaw + wobbling pitch/roll.
        """
        
        vx, vy, vz = 0.0, 0.0, 0.0
        
        # Compute target orientation as quaternion
        target_quat = self.compute_target_orientation(phase)
        
        # Compute angular velocity needed to reach target
        quat_diff = quat_multiply(target_quat, quaternion_conjugate(self.root_quat))
        
        # Extract angular velocity from quaternion difference
        # For small rotations: omega ≈ 2 * [x, y, z] / dt
        angle = 2 * np.arccos(np.clip(quat_diff[0], -1.0, 1.0))
        if angle > 1e-6:
            axis = quat_diff[1:] / np.sin(angle / 2)
            omega_body = axis * angle / max(dt, 1e-6)
        else:
            omega_body = np.zeros(3)
        
        # Transform angular velocity to world frame
        rot_matrix = quaternion_to_rotation_matrix(self.root_quat)
        self.omega_world = rot_matrix @ omega_body
        
        # Clamp angular velocities for stability
        max_omega = 3.0
        self.omega_world = np.clip(self.omega_world, -max_omega, max_omega)
        
        self.vel_world = np.array([vx, vy, vz])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_target_orientation(self, phase):
        """
        Compute target orientation quaternion for current phase.
        """
        # Pitch oscillation (two full cycles per motion cycle)
        pitch = self.pitch_amp * np.sin(4 * np.pi * phase)
        
        # Roll oscillation (quadrant pattern)
        if phase < 0.5:
            roll = self.roll_amp * np.sin(2 * np.pi * phase)
        else:
            roll = -self.roll_amp * np.sin(2 * np.pi * (phase - 0.5))
        
        # Yaw accumulation
        yaw = 2 * np.pi * phase
        
        # Construct quaternion: yaw -> pitch -> roll (ZYX convention)
        quat_yaw = euler_to_quaternion(0, 0, yaw)
        quat_pitch = euler_to_quaternion(0, pitch, 0)
        quat_roll = euler_to_quaternion(roll, 0, 0)
        
        target_quat = quaternion_multiply(quat_yaw, quaternion_multiply(quat_pitch, quat_roll))
        return target_quat

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame maintaining ground contact.
        
        Strategy: Transform nominal world-frame foot position (with adjustments)
        into current body frame to naturally account for all orientation effects.
        """
        
        # Start with nominal world position for this foot
        foot_world = self.nominal_feet_world[leg_name].copy()
        
        # Apply small adjustments in world frame to track wobble
        is_front = leg_name.startswith('F')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Smooth phase-based adjustment factors
        pitch_factor = np.sin(4 * np.pi * phase)
        if phase < 0.5:
            roll_factor = np.sin(2 * np.pi * phase)
        else:
            roll_factor = -np.sin(2 * np.pi * (phase - 0.5))
        
        # Compute tilt magnitude for scaling
        current_pitch = self.pitch_amp * np.sin(4 * np.pi * phase)
        if phase < 0.5:
            current_roll = self.roll_amp * np.sin(2 * np.pi * phase)
        else:
            current_roll = -self.roll_amp * np.sin(2 * np.pi * (phase - 0.5))
        
        tilt_magnitude = np.sqrt(current_pitch**2 + current_roll**2)
        max_tilt = np.sqrt(2) * self.pitch_amp
        adjustment_scale = 1.0 - 0.5 * (tilt_magnitude / max_tilt)**2
        
        # World-frame adjustments (in world coordinates)
        # Get current body orientation to determine world X and Y directions
        rot_matrix = quaternion_to_rotation_matrix(self.root_quat)
        body_x_in_world = rot_matrix[:, 0]
        body_y_in_world = rot_matrix[:, 1]
        
        # Apply adjustments along body-aligned world directions
        if is_front:
            foot_world[:3] += body_x_in_world * self.foot_adjust_x * pitch_factor * adjustment_scale
        else:
            foot_world[:3] -= body_x_in_world * self.foot_adjust_x * pitch_factor * adjustment_scale
        
        if is_left:
            foot_world[:3] += body_y_in_world * self.foot_adjust_y * roll_factor * adjustment_scale
        else:
            foot_world[:3] -= body_y_in_world * self.foot_adjust_y * roll_factor * adjustment_scale
        
        # Add yaw-tracking circular adjustment
        yaw_phase = phase * 2 * np.pi
        yaw_scale = adjustment_scale * 0.8
        if leg_name.startswith('FL'):
            foot_world[0] += self.yaw_adjust * np.sin(yaw_phase) * yaw_scale
            foot_world[1] += self.yaw_adjust * np.cos(yaw_phase) * yaw_scale
        elif leg_name.startswith('FR'):
            foot_world[0] += self.yaw_adjust * np.sin(yaw_phase + np.pi/2) * yaw_scale
            foot_world[1] += self.yaw_adjust * np.cos(yaw_phase + np.pi/2) * yaw_scale
        elif leg_name.startswith('RL'):
            foot_world[0] += self.yaw_adjust * np.sin(yaw_phase + np.pi) * yaw_scale
            foot_world[1] += self.yaw_adjust * np.cos(yaw_phase + np.pi) * yaw_scale
        elif leg_name.startswith('RR'):
            foot_world[0] += self.yaw_adjust * np.sin(yaw_phase + 3*np.pi/2) * yaw_scale
            foot_world[1] += self.yaw_adjust * np.cos(yaw_phase + 3*np.pi/2) * yaw_scale
        
        # Ensure Z is at ground level in world frame
        foot_world[2] = 0.0
        
        # Transform world position to body frame
        foot_relative_world = foot_world - self.root_pos
        rot_matrix_inv = rot_matrix.T
        foot_body = rot_matrix_inv @ foot_relative_world
        
        return foot_body


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion.
    Convention:
      - roll  about X
      - pitch about Y
      - yaw   about Z
    Quaternion format: [w, x, y, z]
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return np.array([w, x, y, z])


def quaternion_multiply(q1, q2):
    """
    Quaternion multiplication.
    q = q1 ⊗ q2
    Both quaternions are [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to rotation matrix (body -> world).
    """
    w, x, y, z = q

    # Normalize for safety
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

    return R


def quaternion_conjugate(q):
    """
    Quaternion conjugate (inverse for unit quaternions).
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])
