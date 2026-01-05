from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SKATE_5_MotionGenerator(BaseMotionGenerator):
    """
    Continuous slalom skating motion.

    - Base translates forward at constant speed.
    - During glide sub‑phases the base oscillates laterally:
      rightward shift during 0.25–0.5, leftward shift during 0.75–1.
    - Leg motions:
      * Left legs (FL, RL) push diagonally backward‑right during 0–0.25.
      * Right legs (FR, RR) push diagonally backward‑left during 0.5–0.75.
      * All legs glide (flat foot, no lift) during the other sub‑phases.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        # Call base constructor
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Motion parameters
        self.forward_speed = 0.5          # m/s forward
        self.lateral_amp = 0.5           # m lateral shift amplitude
        self.step_length = 0.15           # forward/backward push distance
        self.step_height = 0.15           # foot clearance during push

        # Phase offsets for diagonal pushes
        self.phase_offsets = {
            "FL": 0.0,
            "RL": 0.0,
            "FR": 0.5,
            "RR": 0.5,
        }

    # ------------------------------------------------------------------
    # BASE MOTION
    # ------------------------------------------------------------------
    def update_base_motion(self, phase, dt):
        """
        Forward motion with lateral oscillation during glide sub‑phases.
        """
        # Forward velocity
        vx = self.forward_speed

        # Lateral shift during glide phases
        if 0.25 <= phase < 0.5:
            # rightward shift: sinusoidal from 0 to +lateral_amp
            t_rel = (phase - 0.25) / 0.25
            vy = self.lateral_amp * np.sin(np.pi * t_rel)
        elif 0.75 <= phase < 1.0:
            # leftward shift: sinusoidal from 0 to -lateral_amp
            t_rel = (phase - 0.75) / 0.25
            vy = -self.lateral_amp * np.sin(np.pi * t_rel)
        else:
            vy = 0.0

        self.vel_world = np.array([vx, vy, 0.0])
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
        Compute foot target in body frame for the given leg and global phase.
        """
        # Base foot position
        foot = self.base_feet_pos[leg_name].copy()
        is_left = leg_name.startswith("FL") or leg_name.startswith("RL")
        is_right = leg_name.startswith("FR") or leg_name.startswith("RR")

        # Determine sub‑phase for this leg
        if is_left:
            # Left legs: push during 0–0.25
            if 0.0 <= phase < 0.25:
                # Diagonal backward‑right push
                progress = phase / 0.25
                foot[0] -= self.step_length * (progress - 0.5)   # forward/backward
                foot[2] += self.step_height * np.sin(np.pi * progress)  # lift
            else:
                # Glide: flat foot, no vertical motion
                pass
        elif is_right:
            # Right legs: push during 0.5–0.75
            if 0.5 <= phase < 0.75:
                progress = (phase - 0.5) / 0.25
                foot[0] += self.step_length * (progress - 0.5)   # forward/backward
                foot[2] += self.step_height * np.sin(np.pi * progress)  # lift
            else:
                # Glide: flat foot, no vertical motion
                pass

        return foot
