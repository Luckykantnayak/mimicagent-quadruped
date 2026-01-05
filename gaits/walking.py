import numpy as np
from utils.math_utils import *
from gaits.base import BaseMotionGenerator


class WalkingMotionGenerator(BaseMotionGenerator):
    """
    Trot gait with sinusoidal base velocities.

    - Foot trajectories in BODY frame (same trot gait).
    - Base linear velocity vx(t) = sin(2π * f_x * t)
    - Base angular velocity yaw(t) = sin(2π * f_y * t)
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 1.0
        self.duty = 0.75
        self.step_length = 0.1
        self.step_height = 0.08

        # Base foot positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Phase offsets for diagonal trot
        self.phase_offsets = {
            leg_names[0]: 0.0,
            leg_names[1]: 0.5,
            leg_names[2]: 0.5,
            leg_names[3]: 0.0,
        }

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Sinusoidal velocity params
        self.vx_amp = 1.0
        self.yaw_amp = 1.0
        self.vx_freq = 0.5
        self.yaw_freq = 0.2

    def update_base_motion(self, phase, dt):
        """
        Update base using sinusoidal linear velocity in x and yaw rate.
        """
        vx = self.vx_amp * np.sin(2 * np.pi * self.vx_freq * self.t)
        yaw_rate = self.yaw_amp * np.sin(2 * np.pi * self.yaw_freq * self.t)

        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])

        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()

        if leg_phase < self.duty:
            progress = leg_phase / self.duty
            foot[0] -= self.step_length * (progress - 0.5)
        else:
            progress = (leg_phase - self.duty) / (1 - self.duty)
            angle = np.pi * progress
            foot[0] += self.step_length * (progress - 0.5)
            foot[2] += self.step_height * np.sin(angle)

        return foot


# import numpy as np
# from utils.math_utils import *
# from gaits.base import BaseMotionGenerator
#
# # =====================================================
# #           TROT WALKING MOTION GENERATOR
# # =====================================================
#
# class WalkingMotionGenerator(BaseMotionGenerator):
#     """
#     Trot walking gait.
#
#     BASE VELOCITY COMMANDS
#     ----------------------
#     - Linear velocity:  vx=1.0, vy=0.0, vz=0.0  (forward)
#     - Angular velocity: roll_rate=0.0, pitch_rate=0.0, yaw_rate=1.0  (yaw rotation)
#
#     FOOT TRAJECTORIES
#     -----------------
#     - Foot positions are computed in BODY frame
#     - Phase offsets define diagonal trot pattern
#     """
#
#     def __init__(self, initial_foot_positions_body, leg_names,
#                  freq=1.0, duty_ratio=0.75, step_length=0.1, step_height=0.05):
#         super().__init__(initial_foot_positions_body, freq)
#
#         self.leg_names = leg_names
#         self.duty = duty_ratio
#         self.step_length = step_length
#         self.step_height = step_height
#
#         # Base foot positions (BODY frame)
#         self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
#
#         # Phase offsets for diagonal trot
#         # Normalized to [0,1]
#         self.phase_offsets = {
#             leg_names[0]: 0.0,   # FL
#             leg_names[1]: 0.5,   # FR
#             leg_names[2]: 0.5,   # RL
#             leg_names[3]: 0.0    # RR
#         }
#
#         # Set default velocity commands
#         self.set_velocity_command(1.0, 0.0, 0.0)
#         self.set_angular_velocity_command(0.0, 0.0, 1.0)
#
#     def compute_foot_position_body_frame(self, leg_name, phase):
#         """
#         Compute foot position for a single leg in BODY frame based on trot gait.
#
#         Args:
#             leg_name : str
#             phase : float  (normalized scalar in [0,1) for the current cycle)
#
#         Returns:
#             np.ndarray (3,) foot position in BODY frame
#         """
#         # Apply leg-specific phase offset
#         leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
#
#         # Start from base foot position
#         foot = self.base_feet_pos_body[leg_name].copy()
#
#         if leg_phase < self.duty:
#             # Stance: foot moves backward
#             progress = leg_phase / self.duty
#             foot[0] -= self.step_length * (progress - 0.5)
#         else:
#             # Swing: foot lifts and moves forward
#             progress = (leg_phase - self.duty) / (1.0 - self.duty)
#             angle = np.pi * progress
#             foot[0] += self.step_length * (progress - 0.5)
#             foot[2] += self.step_height * np.sin(angle)
#
#         return foot

