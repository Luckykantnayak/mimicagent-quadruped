from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SERPENTINE_BACKWARD_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Dynamic serpentine backward drift motion generator.

    The robot moves backward in a serpentine S-pattern by:
    - Maintaining constant backward velocity (negative vx)
    - Oscillating yaw rate to create body rotation (right then left)
    - Oscillating lateral velocity (left then right) synchronized with yaw
    - Alternately lifting left/right leg pairs to produce dynamic drift
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 0.5  # Cycle frequency for smooth serpentine motion

        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Motion parameters
        self.backward_speed = -0.5  # Constant backward velocity (negative vx)
        self.lateral_drift_amp = 0.3  # Amplitude of lateral velocity oscillation
        self.yaw_rate_amp = 0.8  # Amplitude of yaw rate oscillation

        # Leg movement parameters
        self.lateral_extension_amp = 0.08  # Lateral sway
        self.backward_offset_amp = 0.04   # Slight backward offset during leg swing
        self.z_lift_amp = 0.08            # Vertical lift for dynamic drift

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Velocity commands (world frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with constant backward velocity and oscillating lateral/yaw.
        """
        # Constant backward velocity
        vx = self.backward_speed

        # Lateral velocity oscillation (sinusoidal, phase-shifted for serpentine)
        vy = -self.lateral_drift_amp * np.cos(2 * np.pi * phase)

        # No vertical velocity
        vz = 0.0

        # Yaw rate oscillation
        yaw_rate = self.yaw_rate_amp * np.sin(2 * np.pi * phase)

        # No roll or pitch rate
        roll_rate = 0.0
        pitch_rate = 0.0

        # Set velocity commands in world frame
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])

        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with alternating lateral extensions
        and vertical leg lift for drift.
        """
        foot = self.base_feet_pos_body[leg_name].copy()

        # Define leg groups
        left_legs = ['FL', 'RL']
        right_legs = ['FR', 'RR']

        # Leg lift pattern: left legs lift in [0.0-0.5], right legs lift in [0.5-1.0]
        if any(leg_name.startswith(l) for l in left_legs):
            if phase < 0.5:
                # Lifting left legs
                progress = (phase % 0.5) / 0.5
                foot[2] += self.z_lift_amp * self._smooth_step(progress)
            else:
                # Landing left legs
                progress = ((phase - 0.5) % 0.5) / 0.5
                foot[2] += self.z_lift_amp * (1.0 - self._smooth_step(progress))

        elif any(leg_name.startswith(l) for l in right_legs):
            if phase >= 0.5:
                # Lifting right legs
                progress = ((phase - 0.5) % 0.5) / 0.5
                foot[2] += self.z_lift_amp * self._smooth_step(progress)
            else:
                # Landing right legs
                progress = (phase % 0.5) / 0.5
                foot[2] += self.z_lift_amp * (1.0 - self._smooth_step(progress))

        # Add lateral sway for drift
        lateral_factor = np.sin(2 * np.pi * phase) * self.lateral_extension_amp
        if leg_name in left_legs:
            foot[1] -= lateral_factor
            foot[0] -= self.backward_offset_amp * 0.5
        else:
            foot[1] += lateral_factor
            foot[0] -= self.backward_offset_amp * 0.5

        return foot

    def _smooth_step(self, x):
        """
        Smoothstep function for smooth transitions.
        Maps [0,1] -> [0,1] with zero derivatives at endpoints.
        """
        return x * x * (3.0 - 2.0 * x)
