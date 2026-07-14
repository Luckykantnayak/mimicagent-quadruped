
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
       - End-effector trajectories are expressed in BODY frame
       - Generated per end-effector, keyed by its name (`self.ee_names`)

    5. JOINT OVERRIDES (OPTIONAL)
       - step() returns a "joint_overrides" dict {joint_name: angle_radians}
       - Joints listed there are set directly instead of being solved by IK
       - Use this for joints that are NOT part of an IK chain
         (e.g. a humanoid's arms, torso, or neck)

    ----------------------------------------------------------------------
    AVAILABLE SKILL FUNCTIONS
    ----------------------------------------------------------------------

    The following functions are available to be defined or used by a skill:

        - compute_ee_position_body_frame(ee_name, phase)
        - compute_phase(t)
        - update_base_motion(phase, dt)
        - set_velocity_command(...)
        - set_angular_velocity_command(...)

    The skill decides how (or whether) to use them.

    ----------------------------------------------------------------------
    FRAME CONVENTIONS
    ----------------------------------------------------------------------

    - End-effector positions are ALWAYS expressed in BODY frame
    - Base pose is ALWAYS expressed in WORLD frame

    ======================================================================
    """

    # ==========================================================
    # INITIALIZATION
    # ==========================================================
    def __init__(self, init_ee_pos, freq=1.0):
        """
        Args:
            init_ee_pos : dict[str, np.ndarray]
                Nominal end-effector positions in BODY frame,
                keyed by end-effector name.
            freq : float
                Nominal motion frequency (Hz).
        """
        self.freq = freq

        # End-effector reference positions (BODY frame)
        self.init_ee_pos = {
            k: v.copy() for k, v in init_ee_pos.items()
        }
        self.base_ee_pos = {
            k: v.copy() for k, v in init_ee_pos.items()
        }

        # End-effector list
        self.ee_names = list(init_ee_pos.keys())

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
            3. Compute end-effector targets in BODY frame
            4. Transform end-effector targets to WORLD frame
        """

        phase = self.compute_phase(self.t)

        self.update_base_motion(phase, dt)

        ee_body = self.compute_all_ee_positions_body_frame(phase)
        ee_world = self.compute_all_ee_positions_world_frame(ee_body)

        self.t += dt

        return {
            "time": self.t,
            "phase": phase,
            "root_pos": self.root_pos.copy(),
            "root_quat": self.root_quat.copy(),
            "ee_positions_body": ee_body,
            "ee_positions_world": ee_world,
            "joint_overrides": self.compute_joint_overrides(phase),
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
    # END-EFFECTOR MOTION (BODY FRAME)
    # ==========================================================
    def compute_all_ee_positions_body_frame(self, phase):
        """
        Compute BODY-frame targets for all end-effectors.
        """
        return {
            ee: self.compute_ee_position_body_frame(ee, phase)
            for ee in self.ee_names
        }

    @abstractmethod
    def compute_ee_position_body_frame(self, ee_name, phase):
        """
        Compute BODY-frame target for a single end-effector.

        Args:
            ee_name : str
            phase : float in [0,1]

        Returns:
            np.ndarray (3,)
        """
        pass

    # ==========================================================
    # JOINT OVERRIDES (OPTIONAL)
    # ==========================================================
    def compute_joint_overrides(self, phase):
        """
        Return a dict {joint_name: angle_radians} for joints that should be
        set directly instead of being solved by IK (e.g. arms/torso on a
        humanoid). Default: no overrides.
        """
        return {}

    # ==========================================================
    # WORLD TRANSFORM
    # ==========================================================
    def compute_all_ee_positions_world_frame(self, ee_body):
        """
        Transform BODY-frame end-effector positions to WORLD frame.
        """
        return {
            ee: body_to_world_position(pos, self.root_pos, self.root_quat)
            for ee, pos in ee_body.items()
        }
