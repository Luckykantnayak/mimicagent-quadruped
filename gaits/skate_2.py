from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_SKATE_POWER_SLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Power-slide skating motion generator.
    Implements continuous phase-based kinematic plan with yaw-driven propulsion
    and alternating lateral lock/glide leg motions.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        """
        Parameters
        ----------
        initial_foot_positions_body : dict
            Mapping leg name -> 3D position in BODY frame.
        """
        # Call base initializer
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Store base foot positions
        self.base_feet_pos_body = {
            k: v.copy() for k, v in initial_foot_positions_body.items()
        }

        # Yaw motion parameters
        self.yaw_amp = 0.5          # rad/s amplitude during active phases
        self.yaw_freq = 1.0         # Hz

        # Forward drift parameters (minimal translation)
        self.vx_amp = 0.6           # m/s amplitude
        self.vx_freq = 1.0          # Hz

        self.lock_y = 0.12

    def update_base_motion(self, phase, dt):
        """
        Update base pose using sinusoidal yaw during active subphases
        and minimal forward drift.
        """
        # Determine if phase is in active yaw ranges
        active_yaw = (0.0 <= phase < 0.3) or (0.4 <= phase < 0.7)

        if active_yaw:
            yaw_rate = self.yaw_amp * np.sin(2 * np.pi * self.yaw_freq * self.t)
        else:
            yaw_rate = 0.0

        # Minimal forward drift
        vx = self.vx_amp * (0.5 + 0.5 * abs(yaw_rate) / self.yaw_amp)

        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])

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
        Compute foot target in BODY frame based on current phase and leg role.
        Lateral lock: maintain constant lateral offset (y).
        Glide: foot follows base motion, no change from initial position.
        Re-center: move lateral offset to zero.
        """
        # Base foot reference
        foot = self.base_feet_pos_body[leg_name].copy()

        # Determine subphase
        if 0.0 <= phase < 0.3:
            subphase = "first"
        elif 0.3 <= phase < 0.4:
            subphase = "second"
        elif 0.4 <= phase < 0.7:
            subphase = "third"
        else:  # 0.7 <= phase < 1.0
            subphase = "fourth"

        # Identify left vs right legs using naming convention
        is_left = leg_name.endswith("L") or "_L" in leg_name or leg_name.startswith("FL") or leg_name.startswith("RL")
        is_right = leg_name.endswith("R") or "_R" in leg_name or leg_name.startswith("FR") or leg_name.startswith("RR")

        if subphase == "first":
            # Left legs lock, right legs glide
            if is_left:
                pass  # lateral lock: keep y as is
            elif is_right:
                pass  # glide: no change

        elif subphase == "third":
            # Right legs lock, left legs glide
            if is_right:
                pass  # lateral lock
            elif is_left:
                pass  # glide

        else:
            # Re-center phase
            foot[1] = 0.0

        return foot

