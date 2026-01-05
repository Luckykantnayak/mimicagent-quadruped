from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_WAVE_CRAWLER_MotionGenerator(BaseMotionGenerator):
    """
    Continuous sinusoidal wave crawler.

    - Base follows a sinusoidal forward pitch and vertical undulation.
    - Leg lifts and pushes are phase‑shifted across legs.
    - Foot trajectories are expressed in BODY frame and updated each step.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        # Call base constructor
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Gait parameters
        self.duty = 1.0                     # continuous motion
        self.step_length = 0.08             # forward push amplitude (m)
        self.step_height = 0.06             # lift amplitude (m)

        # Base motion parameters
        self.pitch_amp = 0.1                # radians
        self.vertical_amp = 0.05            # meters

        # Store leg names
        self.leg_names = leg_names

        # Phase offsets distributed evenly across legs
        n_legs = len(leg_names)
        self.phase_offsets = {
            leg: i / n_legs for i, leg in enumerate(leg_names)
        }

    def update_base_motion(self, phase, dt):
        """
        Update base pose with sinusoidal pitch and vertical motion.
        """
        pitch = self.pitch_amp * np.sin(2 * np.pi * phase)
        vertical = self.vertical_amp * np.sin(2 * np.pi * phase)

        # Set base linear velocity in x (forward) and z (vertical)
        self.vel_world = np.array([0.2, 0.0, vertical])

        # Set base angular velocity around y (pitch)
        self.omega_world = np.array([0.0, pitch, 0.0])

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
        Compute foot target in BODY frame with safe lift only during push.
        """
        local_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_init_feet_pos[leg_name].copy()

        # Define step phase (e.g., first half of cycle is push)
        step_fraction = 0.5
        if local_phase < step_fraction:
            progress = local_phase / step_fraction
            foot[0] += self.step_length * np.sin(np.pi * progress)  # forward/backward push
            foot[2] += self.step_height * np.sin(np.pi * progress)  # lift
        else:
            foot[2] = self.base_init_feet_pos[leg_name][2]  # keep foot on ground

        return foot

