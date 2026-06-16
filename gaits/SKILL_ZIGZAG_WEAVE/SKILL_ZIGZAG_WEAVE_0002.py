from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ZIGZAG_WEAVE_MotionGenerator(BaseMotionGenerator):
    """
    Zigzag weaving motion with continuous four-point contact.
    
    The robot performs alternating sharp lateral cuts (right-left-right)
    combined with continuous forward velocity to create a zigzag trajectory.
    All four feet remain in contact throughout to support rapid direction changes.
    
    Phase structure:
      [0.0, 0.2]: Right cut 1 - rightward lateral velocity + positive yaw rate
      [0.2, 0.3]: Stabilize 1 - pure forward motion
      [0.3, 0.5]: Left cut - leftward lateral velocity + negative yaw rate
      [0.5, 0.6]: Stabilize 2 - pure forward motion
      [0.6, 0.8]: Right cut 2 (amplified) - increased rightward velocity + yaw rate
      [0.8, 1.0]: Final stabilize - smooth return to neutral
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # ~2 seconds per full zigzag cycle
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity parameters
        self.vx_forward = 0.8  # Constant forward velocity
        self.vy_cut_base = 0.5  # Base lateral velocity for cuts
        self.vy_cut_amplified = 0.75  # Amplified lateral velocity for second right cut
        self.yaw_rate_base = 1.2  # Base yaw rate for cuts (rad/s)
        self.yaw_rate_amplified = 1.8  # Amplified yaw rate for second right cut
        
        # Stance modulation parameters
        self.lateral_extension = 0.06  # Max lateral foot extension during push phases
        self.lateral_tuck = 0.04  # Lateral tuck amount during opposite cuts
        self.longitudinal_shift = 0.03  # Small forward/back shift during cuts

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        All velocities are in WORLD frame.
        """
        vx = self.vx_forward
        vy = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.2: Right cut 1
        if phase < 0.2:
            vy = self.vy_cut_base
            yaw_rate = self.yaw_rate_base
        
        # Phase 0.2-0.3: Stabilize 1 (smooth ramp down)
        elif phase < 0.3:
            local_phase = (phase - 0.2) / 0.1
            vy = self.vy_cut_base * (1.0 - local_phase)
            yaw_rate = self.yaw_rate_base * (1.0 - local_phase)
        
        # Phase 0.3-0.5: Left cut
        elif phase < 0.5:
            vy = -self.vy_cut_base
            yaw_rate = -self.yaw_rate_base
        
        # Phase 0.5-0.6: Stabilize 2 (smooth ramp down)
        elif phase < 0.6:
            local_phase = (phase - 0.5) / 0.1
            vy = -self.vy_cut_base * (1.0 - local_phase)
            yaw_rate = -self.yaw_rate_base * (1.0 - local_phase)
        
        # Phase 0.6-0.8: Right cut 2 (amplified)
        elif phase < 0.8:
            vy = self.vy_cut_amplified
            yaw_rate = self.yaw_rate_amplified
        
        # Phase 0.8-1.0: Final stabilize (smooth ramp down)
        else:
            local_phase = (phase - 0.8) / 0.2
            vy = self.vy_cut_amplified * (1.0 - local_phase)
            yaw_rate = self.yaw_rate_amplified * (1.0 - local_phase)
        
        # Set velocity commands (WORLD frame)
        self.vel_world = np.array([vx, vy, 0.0])
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
        Compute foot position in BODY frame for given leg and phase.
        
        During right cuts: left legs (FL, RL) extend leftward, right legs (FR, RR) tuck inward
        During left cuts: right legs extend rightward, left legs tuck inward
        All feet remain in contact (z stays at base height)
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front_leg = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        dy = 0.0  # Lateral shift in body frame
        dx = 0.0  # Longitudinal shift
        
        # Phase 0.0-0.2: Right cut 1
        if phase < 0.2:
            progress = phase / 0.2
            if is_left_leg:
                # Left legs extend leftward (negative y in body frame)
                dy = -self.lateral_extension * progress
                dx = self.longitudinal_shift * progress if is_front_leg else -self.longitudinal_shift * progress
            else:
                # Right legs tuck inward (negative y in body frame)
                dy = -self.lateral_tuck * progress
        
        # Phase 0.2-0.3: Stabilize 1
        elif phase < 0.3:
            progress = (phase - 0.2) / 0.1
            if is_left_leg:
                # Return from extended to neutral
                dy = -self.lateral_extension * (1.0 - progress)
                dx = self.longitudinal_shift * (1.0 - progress) if is_front_leg else -self.longitudinal_shift * (1.0 - progress)
            else:
                # Return from tucked to neutral
                dy = -self.lateral_tuck * (1.0 - progress)
        
        # Phase 0.3-0.5: Left cut
        elif phase < 0.5:
            progress = (phase - 0.3) / 0.2
            if is_left_leg:
                # Left legs tuck inward (positive y in body frame)
                dy = self.lateral_tuck * progress
            else:
                # Right legs extend rightward (positive y in body frame)
                dy = self.lateral_extension * progress
                dx = self.longitudinal_shift * progress if is_front_leg else -self.longitudinal_shift * progress
        
        # Phase 0.5-0.6: Stabilize 2
        elif phase < 0.6:
            progress = (phase - 0.5) / 0.1
            if is_left_leg:
                # Return from tucked to neutral
                dy = self.lateral_tuck * (1.0 - progress)
            else:
                # Return from extended to neutral
                dy = self.lateral_extension * (1.0 - progress)
                dx = self.longitudinal_shift * (1.0 - progress) if is_front_leg else -self.longitudinal_shift * (1.0 - progress)
        
        # Phase 0.6-0.8: Right cut 2 (amplified)
        elif phase < 0.8:
            progress = (phase - 0.6) / 0.2
            amplification = 1.3  # Increased amplitude for second right cut
            if is_left_leg:
                # Left legs extend leftward with amplification
                dy = -self.lateral_extension * amplification * progress
                dx = self.longitudinal_shift * amplification * progress if is_front_leg else -self.longitudinal_shift * amplification * progress
            else:
                # Right legs tuck inward with amplification
                dy = -self.lateral_tuck * amplification * progress
        
        # Phase 0.8-1.0: Final stabilize
        else:
            progress = (phase - 0.8) / 0.2
            amplification = 1.3
            if is_left_leg:
                # Return from amplified extended to neutral
                dy = -self.lateral_extension * amplification * (1.0 - progress)
                dx = self.longitudinal_shift * amplification * (1.0 - progress) if is_front_leg else -self.longitudinal_shift * amplification * (1.0 - progress)
            else:
                # Return from amplified tucked to neutral
                dy = -self.lateral_tuck * amplification * (1.0 - progress)
        
        foot[0] += dx
        foot[1] += dy
        # foot[2] remains unchanged (continuous ground contact)
        
        return foot