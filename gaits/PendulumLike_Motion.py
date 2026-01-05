from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class PendulumLike_MotionGenerator(BaseMotionGenerator):
    """
    Skill SKILL_6: Continuous pendulum-like motion.
    - Base oscillates vertically (sinusoidal).
    - FL & RR swing forward during 0–0.3, hold high 0.3–0.6, return low 0.6–1.
    - FR & RL swing backward during 0.3–0.6, hold low otherwise.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        # Call base initializer
        super().__init__(initial_foot_positions_body, freq=2.0)

        # Motion parameters
        self.freq = 2.0                     # base vertical oscillation frequency (Hz)
        self.base_amp_z = 0.1               # amplitude of vertical motion
        self.step_length = 0.35             # horizontal swing distance
        self.step_height = 0.10           # vertical foot lift (unused but kept for consistency)

        # Store initial foot positions
        self.base_feet_pos_body = {
            k: v.copy() for k, v in initial_foot_positions_body.items()
        }

    # ------------------------------------------------------------------
    # BASE MOTION
    # ------------------------------------------------------------------
    def update_base_motion(self, phase, dt):
        """
        Vertical sinusoidal base motion.
        """
        # Current time from internal counter
        t = self.t

        # Desired vertical velocity (derivative of sin)
        vz = 2 * np.pi * self.freq * self.base_amp_z * np.cos(2 * np.pi * self.freq * t)

        # No horizontal or angular motion
        self.vel_world = np.array([0.0, 0.0, vz])
        self.omega_world = np.zeros(3)

        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    # ------------------------------------------------------------------
    # LEG MOTION (BODY FRAME)
    # ------------------------------------------------------------------
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Piecewise foot trajectory based on sub‑phase ranges.
        """
        # Start from the base (rest) position
        foot = self.base_feet_pos_body[leg_name].copy()

        # Helper to add horizontal offset
        def add_offset(offset):
            foot[0] += offset

        # Forward swing legs (FL, RR)
        if leg_name in ["FL", "RR"]:
            if phase < 0.3:
                # Swing forward from low to high
                progress = phase / 0.3
                offset = self.step_length * (progress - 0.5)
                add_offset(offset)
            elif phase < 0.6:
                # Hold at high position
                add_offset(self.step_length * 0.5)
            else:
                # Return to low position
                progress = (phase - 0.6) / 0.4
                offset = self.step_length * (progress - 0.5)
                add_offset(offset)

        # Backward swing legs (FR, RL)
        else:  # "FR" or "RL"
            if phase < 0.3:
                # Hold at low position (no offset)
                pass
            elif phase < 0.6:
                # Swing backward from high to low
                progress = (phase - 0.3) / 0.3
                offset = -self.step_length * (progress - 0.5)
                add_offset(offset)
            else:
                # Hold at low position
                pass

        return foot
