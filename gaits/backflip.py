from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_BACKFLIP_1_MotionGenerator(BaseMotionGenerator):
    """
    Symmetric vertical jump with backward pitch rotation.
    Foot trajectories are defined in BODY frame and follow the sub‑phase
    schedule from the motion plan.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        """
        Parameters
        ----------
        initial_foot_positions_body : dict
            Mapping leg name -> 3‑D position in BODY frame at rest.
        """
        # Call base constructor to set up foot references and time
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Phase offsets are not needed; all legs move synchronously
        self.phase_offsets = {leg: 0.0 for leg in self.leg_names}

        # Physical parameters (chosen to produce a realistic backflip)
        self.crouch_depth = 0.15          # meters
        self.takeoff_height = 0.25       # meters above crouch
        self.tuck_angle_max = -np.deg2rad(45)  # pitch rotation during aerial_tuck
        self.extend_height = 0.05        # additional extension after tuck

        # Timing of sub‑phases (in phase units)
        self.sub_phases = {
            "crouch":      (0.00, 0.20),
            "takeoff":     (0.20, 0.30),
            "aerial_tuck": (0.30, 0.60),
            "aerial_extend":(0.60, 0.80),
            "landing":     (0.80, 1.00)
        }

    # ------------------------------------------------------------------
    # BASE MOTION
    # ------------------------------------------------------------------
    def update_base_motion(self, phase, dt):
        """
        Base remains fixed horizontally. During aerial_tuck it pitches
        backward from 0 to self.tuck_angle_max and then returns to 0.
        """
        # Compute desired pitch angle based on phase
        if self.sub_phases["aerial_tuck"][0] <= phase < self.sub_phases["aerial_tuck"][1]:
            # Linear interpolation from 0 to max negative angle
            t_rel = (phase - self.sub_phases["aerial_tuck"][0]) / (
                self.sub_phases["aerial_tuck"][1] - self.sub_phases["aerial_tuck"][0]
            )
            pitch = t_rel * self.tuck_angle_max
        elif self.sub_phases["aerial_extend"][0] <= phase < self.sub_phases["aerial_extend"][1]:
            # Return to 0 linearly
            t_rel = (phase - self.sub_phases["aerial_extend"][0]) / (
                self.sub_phases["aerial_extend"][1] - self.sub_phases["aerial_extend"][0]
            )
            pitch = (1.0 - t_rel) * self.tuck_angle_max
        else:
            pitch = 0.0

        # No horizontal or yaw motion
        self.vel_world = np.array([0.0, 0.0, 0.2])
        self.omega_world = np.array([0.0, pitch, 0.0])

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
        Compute the foot position in BODY frame for a given leg and phase.
        All legs follow identical motion profiles as defined by the spec.
        """
        base_pos = self.base_init_feet_pos[leg_name].copy()

        # Crouch: flex all legs downward
        if self.sub_phases["crouch"][0] <= phase < self.sub_phases["crouch"][1]:
            t_rel = (phase - self.sub_phases["crouch"][0]) / (
                self.sub_phases["crouch"][1] - self.sub_phases["crouch"][0]
            )
            base_pos[2] -= t_rel * self.crouch_depth

        # Takeoff: extend upward to leave ground
        elif self.sub_phases["takeoff"][0] <= phase < self.sub_phases["takeoff"][1]:
            t_rel = (phase - self.sub_phases["takeoff"][0]) / (
                self.sub_phases["takeoff"][1] - self.sub_phases["takeoff"][0]
            )
            base_pos[2] = -self.crouch_depth + t_rel * self.takeoff_height

        # Aerial tuck: legs tuck inward (reduce lateral offset)
        elif self.sub_phases["aerial_tuck"][0] <= phase < self.sub_phases["aerial_tuck"][1]:
            t_rel = (phase - self.sub_phases["aerial_tuck"][0]) / (
                self.sub_phases["aerial_tuck"][1] - self.sub_phases["aerial_tuck"][0]
            )
            # Tuck inward: interpolate x position towards body center
            tuck_progress = t_rel
            base_pos[0] = self.base_init_feet_pos[leg_name][0] * (1.0 - tuck_progress)

        # Aerial extend: legs extend downward to prepare landing
        elif self.sub_phases["aerial_extend"][0] <= phase < self.sub_phases["aerial_extend"][1]:
            t_rel = (phase - self.sub_phases["aerial_extend"][0]) / (
                self.sub_phases["aerial_extend"][1] - self.sub_phases["aerial_extend"][0]
            )
            base_pos[2] = -self.crouch_depth + self.takeoff_height + t_rel * self.extend_height

        # Landing: legs fully extended, return to initial positions
        elif self.sub_phases["landing"][0] <= phase < self.sub_phases["landing"][1]:
            t_rel = (phase - self.sub_phases["landing"][0]) / (
                self.sub_phases["landing"][1] - self.sub_phases["landing"][0]
            )
            base_pos[2] = (-self.crouch_depth + self.takeoff_height +
                           (1.0 - t_rel) * self.extend_height)

        return base_pos
