from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SideFlipKick_MotionGenerator(BaseMotionGenerator):
    """
    Continuous lateral flip with alternating leg kicks.
    Implements the motion plan described in SKILL_SIDE_FLIP_KICK.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        # Call base constructor
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Store leg names
        self.leg_names = leg_names

        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {
            k: v.copy() for k, v in initial_foot_positions_body.items()
        }

        # Motion parameters
        self.roll_start = np.deg2rad(-30.0)   # lean laterally
        self.roll_mid = np.deg2rad(-90.0)     # mid flip
        self.roll_end  = np.deg2rad(-180.0)   # full flip
        self.roll_recover = np.deg2rad(0.0)   # upright

        self.yaw_start = 0.0
        self.yaw_mid   = np.deg2rad(30.0)     # yaw during push
        self.yaw_end   = np.deg2rad(30.0)     # maintain during aerial
        self.yaw_recover = np.deg2rad(0.0)

        # Forward translation during landing
        self.vx_landing = 0.2

        # Extension distances for legs (BODY frame)
        self.forward_ext = 0.15
        self.backward_ext = -0.15

    def update_base_motion(self, phase, dt):
        """
        Update base pose using piecewise linear roll and yaw,
        with a small forward translation during landing.
        """
        # Roll interpolation
        if phase < 0.25:
            roll = self.roll_start + (self.roll_mid - self.roll_start) * (phase / 0.25)
        elif phase < 0.5:
            roll = self.roll_mid + (self.roll_end - self.roll_mid) * ((phase - 0.25) / 0.25)
        elif phase < 0.75:
            roll = self.roll_end
        else:
            roll = self.roll_end + (self.roll_recover - self.roll_end) * ((phase - 0.75) / 0.25)

        # Yaw interpolation
        if phase < 0.25:
            yaw = self.yaw_start
        elif phase < 0.5:
            yaw = self.yaw_start + (self.yaw_mid - self.yaw_start) * ((phase - 0.25) / 0.25)
        elif phase < 0.75:
            yaw = self.yaw_mid + (self.yaw_end - self.yaw_mid) * ((phase - 0.5) / 0.25)
        else:
            yaw = self.yaw_end + (self.yaw_recover - self.yaw_end) * ((phase - 0.75) / 0.25)

        # Forward translation during landing
        if phase >= 0.75:
            vx = self.vx_landing
        else:
            vx = 0.0

        # Set velocity commands (WORLD frame)
        self.set_velocity_command(vx, 0.0, 0.0)
        self.set_angular_velocity_command(roll, 0.0, yaw)

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
        Compute foot target in BODY frame based on the current phase.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()

        # Determine leg group
        if leg_name.startswith(("FL", "RL")):
            side = "left"
        else:
            side = "right"

        # Phase segments
        if phase < 0.25:  # Preparation
            # Slight flex: lower z a bit
            base_pos[2] -= 0.02
        elif phase < 0.5:  # Lift & Push
            progress = (phase - 0.25) / 0.25
            if side == "left":
                # Lift forward
                base_pos[0] += self.forward_ext * progress
            else:
                # Push backward
                base_pos[0] += self.backward_ext * progress
            # Raise foot slightly
            base_pos[2] += 0.05 * progress
        elif phase < 0.75:  # Aerial Rotation
            # Maintain extended posture (no change)
            pass
        else:  # Landing & Recovery
            progress = (phase - 0.75) / 0.25
            # Descend
            base_pos[2] -= 0.05 * progress
            # Slight flex on touchdown
            base_pos[2] -= 0.01 * progress

        return base_pos
