from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PROBE_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Slow probing gait with phased foot lifts and base height oscillation.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Use provided leg names (order matters)
        self.leg_names = leg_names

        # Group legs by index instead of hard-coded names
        # group1: leg 0 & 3, group2: leg 1 & 2
        self.phase_offsets = {
            self.leg_names[0]: 0.0,
            self.leg_names[3]: 0.0,
            self.leg_names[1]: 0.25,
            self.leg_names[2]: 0.25,
        }

        # Foot trajectory parameters
        self.step_height = 0.1
        self.small_lift = 0.04

        # Base motion parameters
        self.base_speed = 0.3
        self.base_height = 0.1
        self.height_osc_amp = 0.02


    def update_base_motion(self, phase, dt):
        """
        Base moves forward at constant speed and oscillates vertically.
        Orientation remains fixed (no rotation).
        """
        # Forward velocity in world frame
        self.vel_world = np.array([self.base_speed, 0.0, 0.0])

        # Zero angular velocity
        self.omega_world = np.zeros(3)

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
        Compute foot position in body frame based on phased lift schedule.
        """
        # Reference foot position
        pos = self.base_feet_pos[leg_name].copy()

        # Local phase for this leg
        local_phase = (phase + self.phase_offsets.get(leg_name, 0.0)) % 1.0

        # Probe phase: lift slightly
        if local_phase < 0.25:
            progress = local_phase / 0.25
            lift = self.step_height * np.sin(np.pi * progress)
            pos[2] += lift

        # Settling phase: maintain contact (no change)

        # Forward step phase: hold foot in place

        # Reset phase: small lift before next probe
        elif local_phase >= 0.75:
            progress = (local_phase - 0.75) / 0.25
            lift = self.small_lift * np.sin(np.pi * progress)
            pos[2] += lift

        return pos
