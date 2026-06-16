from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_PACE_MotionGenerator(BaseMotionGenerator):
    """
    Continuous pace gait: left and right side legs move in-phase, alternating every half cycle.
    Base translates forward smoothly; yaw remains minimal.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        # Call base constructor
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Gait parameters
        self.leg_names = leg_names
        self.duty = 0.5                     # stance duration fraction
        self.step_length = 0.12             # forward step length (m)
        self.step_height = 0.07             # swing height (m)

        # Base foot positions in body frame
        self.base_feet_pos_body = {
            k: v.copy() for k, v in initial_foot_positions_body.items()
        }

        # Phase offsets: left side (FL, RL) start at 0, right side (FR, RR) offset by 0.5
        self.phase_offsets = {
            self.leg_names[0]: 0.0,   # FL
            self.leg_names[1]: 0.5,   # FR
            self.leg_names[2]: 0.0,   # RL
            self.leg_names[3]: 0.5    # RR
        }

        # Base motion parameters
        self.vx_amp = 0.4                 # peak forward speed (m/s)
        self.base_freq = 1.0              # base motion frequency

    def update_base_motion(self, phase, dt):
        """
        Base translates forward with velocity peaking during stance phases.
        Velocity follows a cosine shape over the full cycle.
        """
        # Cosine profile: 0 at phase=0.5 (mid swing), max at phase=0 or 1
        vx = self.vx_amp * np.cos(2 * np.pi * phase)
        # Minimal yaw
        yaw_rate = 0.0

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
        """
        Compute foot target in body frame.
        Left side legs swing during [0, 0.5), right side legs swing during [0.5,1).
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()

        if leg_phase < self.duty:
            # Stance: foot stays at base position
            pass
        else:
            # Swing phase: move forward then back to stance
            swing_progress = (leg_phase - self.duty) / (1.0 - self.duty)
            # Forward motion: linear interpolation from behind to forward
            if swing_progress < 0.5:
                # Move forward
                progress = swing_progress / 0.5
                foot[0] -= self.step_length * (progress - 0.5)
            else:
                # Return to stance
                progress = (swing_progress - 0.5) / 0.5
                foot[0] += self.step_length * (progress - 0.5)
            # Vertical lift using sine
            angle = np.pi * swing_progress
            foot[2] += self.step_height * np.sin(angle)

        return foot

    def compute_phase(self, t):
        """
        Override to use continuous phase in [0,1] with specified frequency.
        """
        return (self.base_freq * t) % 1.0
