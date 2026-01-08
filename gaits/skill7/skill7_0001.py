from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_SPIRAL_VAULT_MotionGenerator(BaseMotionGenerator):
    """
    Continuous aerial vault with roll rotation and scissor leg pattern.
    Implements the motion plan described in the planner specification.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        # Call base constructor
        super().__init__(initial_foot_positions_body, freq=1.)

        # Gait parameters
        self.step_height = 0.15          # maximum lift height (m)
        self.flight_roll_rate = 4 * np.pi   # rad/s, full rotation over flight phase

        # Phase boundaries
        self.push_off_start = 0.0
        self.push_off_end   = 0.1
        self.flight_start   = 0.1
        self.flight_end     = 0.6
        self.landing_start  = 0.6
        self.landing_end    = 1.

        # Store initial foot positions for reference
        self.base_feet_pos_body = {
            k: v.copy() for k, v in initial_foot_positions_body.items()
        }

    # ------------------------------------------------------------------
    # Base motion: roll only during flight phase
    # ------------------------------------------------------------------
    def update_base_motion(self, phase, dt):
        if self.flight_start <= phase < self.flight_end:
            # Continuous roll during flight
            self.vel_world = np.zeros(3)
            self.omega_world = np.array([self.flight_roll_rate, 0.0, 0.0])
        else:
            # No motion during push_off and landing
            self.vel_world = np.zeros(3)
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
    # Leg motion: piecewise linear z offsets per leg
    # ------------------------------------------------------------------
    def compute_foot_position_body_frame(self, leg_name, phase):
        base_pos = self.base_feet_pos_body[leg_name].copy()
        z_offset = 0.0

        # Helper to compute linear interpolation
        def lerp(a, b, t):
            return a + (b - a) * t

        # Push-off phase
        if self.push_off_start <= phase < self.push_off_end:
            t = (phase - self.push_off_start) / (self.push_off_end - self.push_off_start)
            if leg_name.startswith("FL"):
                # Lift high then descend to ground
                if t < 0.5:
                    z_offset = lerp(0.0, self.step_height, t * 2)
                else:
                    z_offset = lerp(self.step_height, 0.0, (t - 0.5) * 2)
            elif leg_name.startswith("FR"):
                # Remain on ground
                z_offset = 0.0
            elif leg_name.startswith(("RL", "RR")):
                # Push off then lift slightly (same as FL)
                if t < 0.5:
                    z_offset = lerp(0.0, self.step_height * 0.6, t * 2)
                else:
                    z_offset = lerp(self.step_height * 0.6, 0.0, (t - 0.5) * 2)
        # Flight phase
        elif self.flight_start <= phase < self.flight_end:
            if leg_name.startswith(("FL", "RL")):
                # Maintain scissor position with slight flex
                z_offset = self.step_height * 0.5
            else:
                # Front legs also maintain scissor position
                z_offset = self.step_height * 0.5
        # Landing phase
        elif self.landing_start <= phase < self.landing_end:
            t = (phase - self.landing_start) / (self.landing_end - self.landing_start)
            if leg_name.startswith(("FL", "RL")):
                # Lower to staggered stance
                z_offset = lerp(self.step_height * 0.5, 0.0, t)
            elif leg_name.startswith(("FR", "RR")):
                # Lower to staggered stance
                z_offset = lerp(self.step_height * 0.5, 0.0, t)
        else:
            # Default to ground
            z_offset = 0.0

        base_pos[2] += z_offset
        return base_pos
