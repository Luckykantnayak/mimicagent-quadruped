from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Ricochet bounce diagonal gait with alternating left/right diagonal bounces.
    
    Motion cycle:
    - Phase 0.0-0.2: Left compression with positive yaw
    - Phase 0.2-0.4: Right diagonal launch (aerial)
    - Phase 0.4-0.6: Right compression with negative yaw
    - Phase 0.6-0.8: Left diagonal launch (aerial)
    - Phase 0.8-1.0: Landing preparation with yaw neutralization
    
    Contact pattern alternates between left-side (FL, RL) and right-side (FR, RR) diagonal pairs.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8
        
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Increased nominal height to provide clearance margin
        self.nominal_height = 0.55
        # Increased regulation gain for stronger height control
        self.height_regulation_gain = 2.8
        
        # Reduced vertical velocities to constrain height envelope
        self.forward_velocity = 0.8
        self.base_compression_velocity = -0.42
        self.base_launch_velocity = 0.
        self.launch_lateral_velocity = 0.8
        self.yaw_rate_magnitude = 2.5
        
        # Increased swing clearance and removed stance compression in body frame
        self.swing_clearance = 0.08
        self.forward_reach = 0.08
        self.lateral_spread = 0.03

    def update_base_motion(self, phase, dt):
        vx = self.forward_velocity
        vy = 0.0
        yaw_rate = 0.0

        # Height regulation only (never launch)
        height_error = self.nominal_height - self.root_pos[2]
        vz = np.clip(height_error * self.height_regulation_gain, -0.5, 0.5)

        # Phase-based diagonal compression illusion
        if 0.0 <= phase < 0.2:
            vz += self.base_compression_velocity * np.sin(np.pi * phase / 0.2)
            yaw_rate = self.yaw_rate_magnitude * (1.0 - phase / 0.2)

        elif 0.2 <= phase < 0.4:
            vy = self.launch_lateral_velocity * (1.0 - (phase - 0.2) / 0.2)
            yaw_rate = -self.yaw_rate_magnitude * 0.5

        elif 0.4 <= phase < 0.6:
            t = (phase - 0.4) / 0.2
            vz += self.base_compression_velocity * np.sin(np.pi * t)
            yaw_rate = -self.yaw_rate_magnitude * (1.0 - t)

        elif 0.6 <= phase < 0.8:
            vy = -self.launch_lateral_velocity * (1.0 - (phase - 0.6) / 0.2)
            yaw_rate = self.yaw_rate_magnitude * 0.5

        else:
            yaw_rate *= 0.3  # settle

        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])

        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for each leg based on phase.
        Feet maintain consistent body-frame z-positions during stance.
        Body compression is achieved via base vertical motion, not foot retraction.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Left-side legs: FL, RL (stance during 0.0-0.2 and 0.6-1.0)
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            is_front = leg_name.startswith('FL')
            
            # Phase 0.0-0.2: Stance (feet maintain ground contact, body compresses)
            if 0.0 <= phase < 0.2:
                t_local = phase / 0.2
                # No vertical adjustment in body frame - compression via base motion only
                foot[0] += self.forward_reach * (1.0 - 0.3 * t_local) if is_front else -self.forward_reach * 0.5
                foot[1] -= self.lateral_spread * (1.0 - 0.2 * t_local)
            
            # Phase 0.2-0.4: Swing with clearance
            elif 0.2 <= phase < 0.4:
                t_local = (phase - 0.2) / 0.2
                clearance_profile = np.sin(t_local * np.pi)
                foot[2] += self.swing_clearance * clearance_profile
                foot[0] += self.forward_reach * (0.7 + 0.3 * t_local) if is_front else -self.forward_reach * 0.3
                foot[1] -= self.lateral_spread * (1.0 - t_local * 0.5)
            
            # Phase 0.4-0.6: Continue swing
            elif 0.4 <= phase < 0.6:
                t_local = (phase - 0.4) / 0.2
                clearance_profile = np.sin((0.5 + t_local * 0.5) * np.pi)
                foot[2] += self.swing_clearance * clearance_profile
                foot[0] += self.forward_reach * (1.0 - 0.2 * t_local) if is_front else -self.forward_reach * 0.4
                foot[1] -= self.lateral_spread * 0.5
            
            # Phase 0.6-0.8: Transition to stance (lower foot smoothly to ground)
            elif 0.6 <= phase < 0.8:
                t_local = (phase - 0.6) / 0.2
                # Smooth descent from swing clearance to ground contact
                clearance_profile = np.cos(t_local * np.pi * 0.5)
                foot[2] += self.swing_clearance * clearance_profile * clearance_profile
                foot[0] += self.forward_reach * (0.8 + 0.2 * t_local) if is_front else -self.forward_reach * 0.5
                foot[1] -= self.lateral_spread * (0.5 + 0.5 * t_local)
            
            # Phase 0.8-1.0: Early stance
            else:
                t_local = (phase - 0.8) / 0.2
                # Feet grounded, no vertical adjustment in body frame
                foot[0] += self.forward_reach if is_front else -self.forward_reach * 0.5
                foot[1] -= self.lateral_spread
        
        # Right-side legs: FR, RR (stance during 0.2-0.6)
        else:
            is_front = leg_name.startswith('FR')
            
            # Phase 0.0-0.2: Swing with clearance
            if 0.0 <= phase < 0.2:
                t_local = phase / 0.2
                clearance_profile = np.sin(t_local * np.pi)
                foot[2] += self.swing_clearance * clearance_profile
                foot[0] += self.forward_reach * (0.7 + 0.3 * t_local) if is_front else -self.forward_reach * 0.3
                foot[1] += self.lateral_spread * (1.0 - t_local * 0.5)
            
            # Phase 0.2-0.4: Transition to stance (lower foot smoothly to ground)
            elif 0.2 <= phase < 0.4:
                t_local = (phase - 0.2) / 0.2
                clearance_profile = np.cos(t_local * np.pi * 0.5)
                foot[2] += self.swing_clearance * clearance_profile * clearance_profile
                foot[0] += self.forward_reach * (0.8 + 0.2 * t_local) if is_front else -self.forward_reach * 0.5
                foot[1] += self.lateral_spread * (0.5 + 0.5 * t_local)
            
            # Phase 0.4-0.6: Stance (feet maintain ground contact, body compresses)
            elif 0.4 <= phase < 0.6:
                t_local = (phase - 0.4) / 0.2
                # No vertical adjustment in body frame - compression via base motion only
                foot[0] += self.forward_reach * (1.0 - 0.3 * t_local) if is_front else -self.forward_reach * 0.5
                foot[1] += self.lateral_spread * (1.0 - 0.2 * t_local)
            
            # Phase 0.6-0.8: Swing with clearance
            elif 0.6 <= phase < 0.8:
                t_local = (phase - 0.6) / 0.2
                clearance_profile = np.sin(t_local * np.pi)
                foot[2] += self.swing_clearance * clearance_profile
                foot[0] += self.forward_reach * (0.7 + 0.3 * t_local) if is_front else -self.forward_reach * 0.3
                foot[1] += self.lateral_spread * (1.0 - t_local * 0.5)
            
            # Phase 0.8-1.0: Continue swing
            else:
                t_local = (phase - 0.8) / 0.2
                clearance_profile = np.sin((0.5 + t_local * 0.5) * np.pi)
                foot[2] += self.swing_clearance * clearance_profile
                foot[0] += self.forward_reach * (1.0 - 0.2 * t_local) if is_front else -self.forward_reach * 0.4
                foot[1] += self.lateral_spread * 0.5
        
        return foot