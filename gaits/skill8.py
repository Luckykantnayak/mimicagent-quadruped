from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_WAVE_CRAWLER_FLIP_MotionGenerator(BaseMotionGenerator):
    """
    Continuous wave undulation followed by a forward flip.
    Implements the motion plan described in the planner spec.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        # Call base constructor
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Subphase boundaries (normalized [0,1])
        self.wave_end = 0.30
        self.push_off_start = 0.30
        self.push_off_end   = 0.45
        self.flip_start     = 0.45
        self.flip_end       = 0.70
        self.landing_start  = 0.70

        # Base motion parameters
        self.push_forward_dist = 0.05   # meters forward during push_off
        self.max_pitch_angle   = np.pi  # radians (full flip)

        # Leg motion parameters
        self.scissor_length = 0.2   # forward movement during scissoring
        self.scissor_height = 0.04   # lift during scissoring
        self.swing_length   = 0.10   # forward swing during flip
        self.swing_height   = 0.06   # lift during swing

    def update_base_motion(self, phase, dt):
        """
        Update base pose based on subphase.
        """
        # Base remains stationary during wave_undulation
        if phase < self.wave_end:
            self.vel_world = np.zeros(3)
            self.omega_world = np.zeros(3)

        # Push_off: small forward translation and start pitch
        elif self.push_off_start <= phase < self.push_off_end:
            progress = (phase - self.push_off_start) / (self.push_off_end - self.push_off_start)
            forward = self.push_forward_dist * progress
            pitch_rate = (self.max_pitch_angle / (self.flip_end - self.flip_start)) * progress
            self.vel_world = np.array([forward, 0.0, 0.0])
            self.omega_world = np.array([pitch_rate, 0.0, 0.0])

        # Flip: continuous pitch from upright to inverted and back
        elif self.flip_start <= phase < self.flip_end:
            flip_progress = (phase - self.flip_start) / (self.flip_end - self.flip_start)
            pitch_angle = self.max_pitch_angle * np.sin(np.pi * flip_progress)  # smooth
            pitch_rate = self.max_pitch_angle * np.cos(np.pi * flip_progress) * np.pi / (self.flip_end - self.flip_start)
            self.vel_world = np.array([0.0, 0.0, 0.0])
            self.omega_world = np.array([pitch_rate, 0.0, 0.0])

        # Landing: stop rotation and return to neutral
        else:
            self.vel_world = np.zeros(3)
            self.omega_world = np.array([0.0, 0.0, 0.0])

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
        Compute foot target in body frame based on subphase and leg group.
        """
        base_pos = self.base_init_feet_pos[leg_name].copy()

        # Group 1: FL, FR (scissoring)
        if leg_name.startswith(("FL", "FR")):
            # Scissor forward during wave_undulation
            if phase < self.wave_end:
                progress = phase / self.wave_end
                base_pos[0] += self.scissor_length * (progress - 0.5)
                base_pos[2] += self.scissor_height * np.sin(np.pi * progress)

            # Swing during flip
            if self.flip_start <= phase < self.flip_end:
                progress = (phase - self.flip_start) / (self.flip_end - self.flip_start)
                base_pos[0] += self.swing_length * progress
                base_pos[2] += self.swing_height * np.sin(np.pi * progress)

        # Group 2: RL, RR (push off and stabilize)
        else:
            # Push off during push_off
            if self.push_off_start <= phase < self.push_off_end:
                progress = (phase - self.push_off_start) / (self.push_off_end - self.push_off_start)
                base_pos[0] += self.scissor_length * progress
                base_pos[2] -= self.scissor_height * np.sin(np.pi * progress)

            # Stabilize during landing
            if phase >= self.landing_start:
                base_pos[0] += 0.0
                base_pos[2] -= self.scissor_height * np.sin(np.pi * (phase - self.landing_start) / (1.0 - self.landing_start))

        return base_pos
