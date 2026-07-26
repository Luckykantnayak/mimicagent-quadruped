from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERSE_HELICOPTER_DESCENT_MotionGenerator(BaseMotionGenerator):
    """
    Reverse Helicopter Descent: Robot travels backward while spinning counter-clockwise
    and descending. Legs extend/retract in a rotating cross pattern synchronized with yaw rotation.

    Motion characteristics:
    - Base: constant backward velocity, downward descent, counter-clockwise rotation (360° per cycle)
    - Legs: diagonal pairs alternate between extended and retracted in rotating cross pattern
    - All legs maintain sliding ground contact throughout motion
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        # Initialize base class
        base_init_feet_pos = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        super().__init__(base_init_feet_pos, freq=0.5)

        self.leg_names = leg_names

        # Base motion parameters
        self.vx_backward = -0.3  # Moderate backward velocity (m/s)
        self.vz_descent = -0.15   # Moderate descent rate (m/s)
        self.yaw_rate_ccw = 2.0 * np.pi * self.freq  # 360 degrees per cycle (rad/s)

        # Leg extension parameters
        self.max_radial_extension = 0.12  # Maximum outward extension from base position (m)
        self.min_radial_extension = 0.0   # Minimum extension (retracted to base)

        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Define diagonal pairs and their phase behavior
        # Group 1 (FL, RR): extended [0.0, 0.33], transition to retracted [0.33, 0.67], retracted [0.67, 1.0]
        # Group 2 (FR, RL): retracted [0.0, 0.33], transition to extended [0.33, 0.67], extended [0.67, 1.0]
        self.group_1 = [leg_names[0], leg_names[3]]  # FL, RR
        self.group_2 = [leg_names[1], leg_names[2]]  # FR, RL

    def update_base_motion(self, phase, dt):
        """
        Update base with constant backward velocity, descent, and counter-clockwise yaw rate.
        """
        # Constant velocities throughout all phases
        self.vel_world = np.array([self.vx_backward, 0.0, self.vz_descent])
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate_ccw])

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
        Compute foot position in body frame with rotating cross pattern.

        Phase mapping:
        - [0.0, 0.33]: Group 1 (FL, RR) extended, Group 2 (FR, RL) retracted
        - [0.33, 0.67]: Transition phase - groups swap extension states
        - [0.67, 1.0]: Group 1 retracted, Group 2 extended
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()

        # Compute radial direction from body center to base foot position (in XY plane)
        radial_xy = base_pos[:2].copy()
        radial_distance = np.linalg.norm(radial_xy)

        if radial_distance > 1e-6:
            radial_direction = radial_xy / radial_distance
        else:
            # Fallback for feet at origin
            radial_direction = np.array([1.0, 0.0])

        # Determine extension factor based on leg group and phase
        if leg_name in self.group_1:  # FL, RR
            if phase < 0.33:
                # Extended
                extension_factor = 1.0
            elif phase < 0.67:
                # Transition: extended -> retracted
                transition_phase = (phase - 0.33) / 0.34
                extension_factor = 1.0 - transition_phase
            else:
                # Retracted
                extension_factor = 0.0
        else:  # FR, RL (group_2)
            if phase < 0.33:
                # Retracted
                extension_factor = 0.0
            elif phase < 0.67:
                # Transition: retracted -> extended
                transition_phase = (phase - 0.33) / 0.34
                extension_factor = transition_phase
            else:
                # Extended
                extension_factor = 1.0

        # Smooth extension factor with cosine interpolation
        extension_factor_smooth = 0.5 * (1.0 - np.cos(np.pi * extension_factor))

        # Compute radial extension
        radial_offset = self.max_radial_extension * extension_factor_smooth

        # Apply radial extension in body frame XY plane
        foot_pos = base_pos.copy()
        foot_pos[0] += radial_direction[0] * radial_offset
        foot_pos[1] += radial_direction[1] * radial_offset

        # Z position: maintain ground contact (z=0 in body frame represents ground plane)
        # Feet slide along ground, so keep z at base position (slight adjustment for contact)
        foot_pos[2] = base_pos[2]

        return foot_pos