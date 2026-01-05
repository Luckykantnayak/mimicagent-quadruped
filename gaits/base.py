
from abc import ABC, abstractmethod
import numpy as np
from utils.math_utils import *


class BaseMotionGenerator(ABC):
    """
    Abstract base class for all motion / skill generators.

    1. Time & Phase
       - Internal time variable `t`
       - Scalar phase `phase ∈ [0,1]`
       - Phase representation via `compute_phase(t)`
       - phase = 0 → motion start
       - phase = 1 → motion end

    2. BASE STATE (WORLD FRAME)
       - Base position:    root_pos  ∈ ℝ³
       - Base orientation: root_quat ∈ ℍ  ([w, x, y, z])

    3. BASE COMMAND INTERFACE (WORLD FRAME)
       - Linear velocity command:
            set_velocity_command(vx, vy, vz)
       - Angular velocity command:
            set_angular_velocity_command(roll_rate, pitch_rate, yaw_rate)

    4. END-EFFECTOR TARGETS (BODY FRAME)
       - Foot / end-effector trajectories are expressed in BODY frame
       - Generated per leg

    ----------------------------------------------------------------------
    AVAILABLE SKILL FUNCTIONS
    ----------------------------------------------------------------------

    The following functions are available to be defined or used by a skill:

        - compute_foot_position_body_frame(leg_name, phase)
        - compute_phase(t)
        - update_base_motion(phase, dt)
        - set_velocity_command(...)
        - set_angular_velocity_command(...)

    The skill decides how (or whether) to use them.

    ----------------------------------------------------------------------
    FRAME CONVENTIONS
    ----------------------------------------------------------------------

    - Foot positions are ALWAYS expressed in BODY frame
    - Base pose is ALWAYS expressed in WORLD frame

    ======================================================================
    """

    # ==========================================================
    # INITIALIZATION
    # ==========================================================
    def __init__(self, base_init_feet_pos, freq=1.0):
        """
        Args:
            base_init_feet_pos : dict[str, np.ndarray]
                Nominal foot positions in BODY frame.
            freq : float
                Nominal motion frequency (Hz).
        """
        self.freq = freq

        # Foot reference positions (BODY frame)
        self.base_init_feet_pos = {
            k: v.copy() for k, v in base_init_feet_pos.items()
        }
        self.base_feet_pos = {
            k: v.copy() for k, v in base_init_feet_pos.items()
        }

        # Leg list
        self.leg_names = list(base_init_feet_pos.keys())

        # Time
        self.t = 0.0

        # Base state (WORLD frame)
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Commands (WORLD frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    # ==========================================================
    # RESET
    # ==========================================================
    def reset(self, root_pos, root_quat):
        """
        Reset internal state.

        Args:
            root_pos : np.ndarray (3,)
                Initial base position in WORLD frame.
            root_quat : np.ndarray (4,)
                Initial base orientation [w, x, y, z].
        """
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0

    # ==========================================================
    # COMMAND INTERFACE
    # ==========================================================
    def set_velocity_command(self, vx=0.0, vy=0.0, vz=0.0):
        """
        Set desired base linear velocity in WORLD frame.
        """
        self.vel_world = np.array([vx, vy, vz])

    def set_angular_velocity_command(self, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0):
        """
        Set desired base angular velocity in WORLD frame.
        """
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])

    # ==========================================================
    # MAIN STEP (DO NOT MODIFY)
    # ==========================================================
    def step(self, dt):
        """
        Advance the motion generator by one timestep.

        Pipeline:
            1. Compute phase
            2. Update base pose
            3. Compute foot targets in BODY frame
            4. Transform foot targets to WORLD frame
        """

        phase = self.compute_phase(self.t)

        self.update_base_motion(phase, dt)

        feet_body = self.compute_all_foot_positions_body_frame(phase)
        feet_world = self.compute_all_foot_positions_world_frame(feet_body)

        self.t += dt

        return {
            "time": self.t,
            "phase": phase,
            "root_pos": self.root_pos.copy(),
            "root_quat": self.root_quat.copy(),
            "foot_positions_body": feet_body,
            "foot_positions_world": feet_world,
        }

    # ==========================================================
    # PHASE
    # ==========================================================
    def compute_phase(self, t):
        """
        Compute scalar phase in [0,1].

        Default behavior:
            phase progresses linearly with time based on frequency.
        """
        return (self.freq * t) % 1.0

    # ==========================================================
    # BASE MOTION
    # ==========================================================
    def update_base_motion(self, phase, dt):
        """
        Update base position and orientation.

        Default behavior:
            Integrates commanded WORLD-frame velocities.
        """
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    # ==========================================================
    # FOOT MOTION (BODY FRAME)
    # ==========================================================
    def compute_all_foot_positions_body_frame(self, phase):
        """
        Compute BODY-frame foot targets for all legs.
        """
        return {
            leg: self.compute_foot_position_body_frame(leg, phase)
            for leg in self.leg_names
        }

    @abstractmethod
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute BODY-frame foot target for a single leg.

        Args:
            leg_name : str
            phase : float in [0,1]

        Returns:
            np.ndarray (3,)
        """
        pass

    # ==========================================================
    # WORLD TRANSFORM
    # ==========================================================
    def compute_all_foot_positions_world_frame(self, feet_body):
        """
        Transform BODY-frame foot positions to WORLD frame.
        """
        return {
            leg: body_to_world_position(pos, self.root_pos, self.root_quat)
            for leg, pos in feet_body.items()
        }
