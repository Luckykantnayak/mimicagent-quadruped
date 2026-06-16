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
        self.freq = 0.8  # Cycle frequency for ricochet pattern
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.forward_velocity = 0.8  # Sustained forward velocity
        self.compression_velocity = -1.2  # Downward velocity during compression
        self.launch_vertical_velocity = 2.5  # Upward velocity during launch
        self.launch_lateral_velocity = 1.0  # Lateral velocity during diagonal launch
        self.yaw_rate_magnitude = 3.0  # Yaw rotation rate during compression (rad/s)
        
        # Leg motion parameters
        self.compression_height = 0.12  # Vertical compression distance
        self.extension_height = 0.15  # Vertical extension during aerial
        self.retraction_height = 0.10  # Retraction height during swing
        self.forward_reach = 0.08  # Forward reach during stance preparation
        self.lateral_spread = 0.03  # Lateral foot adjustment during stance

    def update_base_motion(self, phase, dt):
        """
        Update base motion with phase-dependent velocity commands.
        Implements alternating compression/launch pattern with yaw rotation.
        """
        vx = self.forward_velocity
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.2: Left compression with positive yaw
        if 0.0 <= phase < 0.2:
            vz = self.compression_velocity
            yaw_rate = self.yaw_rate_magnitude
        
        # Phase 0.2-0.4: Right diagonal launch (aerial)
        elif 0.2 <= phase < 0.4:
            vz = self.launch_vertical_velocity
            vy = self.launch_lateral_velocity  # Rightward component
            yaw_rate = 0.0
        
        # Phase 0.4-0.6: Right compression with negative yaw
        elif 0.4 <= phase < 0.6:
            vz = self.compression_velocity
            yaw_rate = -self.yaw_rate_magnitude
        
        # Phase 0.6-0.8: Left diagonal launch (aerial)
        elif 0.6 <= phase < 0.8:
            vz = self.launch_vertical_velocity
            vy = -self.launch_lateral_velocity  # Leftward component
            yaw_rate = 0.0
        
        # Phase 0.8-1.0: Landing preparation with yaw neutralization
        else:
            vz = self.compression_velocity * 0.5  # Gentler descent
            yaw_rate = self.yaw_rate_magnitude * 0.5  # Smooth transition back to positive
        
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
        Implements alternating diagonal contact pattern with compression/extension cycles.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Left-side legs: FL, RL
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            is_front = leg_name.startswith('FL')
            
            # Phase 0.0-0.2: Stance with compression
            if 0.0 <= phase < 0.2:
                progress = phase / 0.2
                foot[2] += self.compression_height * (1.0 - progress)
                foot[0] += self.forward_reach if is_front else -self.forward_reach * 0.5
                foot[1] -= self.lateral_spread
            
            # Phase 0.2-0.4: Extension during aerial (right launch)
            elif 0.2 <= phase < 0.4:
                progress = (phase - 0.2) / 0.2
                foot[2] -= self.extension_height * progress
                foot[0] += self.forward_reach * 0.5 if is_front else -self.forward_reach * 0.3
            
            # Phase 0.4-0.6: Retracted during right stance
            elif 0.4 <= phase < 0.6:
                progress = (phase - 0.4) / 0.2
                foot[2] += self.retraction_height
                foot[0] -= self.forward_reach * 0.3 if is_front else -self.forward_reach * 0.2
            
            # Phase 0.6-0.8: Retracted to extended transition (left launch)
            elif 0.6 <= phase < 0.8:
                progress = (phase - 0.6) / 0.2
                retract_factor = 1.0 - progress
                foot[2] += self.retraction_height * retract_factor
                foot[0] += self.forward_reach * progress if is_front else -self.forward_reach * 0.5 * progress
            
            # Phase 0.8-1.0: Landing and early stance
            else:
                progress = (phase - 0.8) / 0.2
                foot[2] += self.compression_height * (1.0 - progress * 0.5)
                foot[0] += self.forward_reach if is_front else -self.forward_reach * 0.5
                foot[1] -= self.lateral_spread
        
        # Right-side legs: FR, RR
        else:
            is_front = leg_name.startswith('FR')
            
            # Phase 0.0-0.2: Retracted during left stance
            if 0.0 <= phase < 0.2:
                progress = phase / 0.2
                foot[2] += self.retraction_height
                foot[0] -= self.forward_reach * 0.3 if is_front else -self.forward_reach * 0.2
            
            # Phase 0.2-0.4: Retracted to extended transition (right launch)
            elif 0.2 <= phase < 0.4:
                progress = (phase - 0.2) / 0.2
                retract_factor = 1.0 - progress
                foot[2] += self.retraction_height * retract_factor
                foot[0] += self.forward_reach * progress if is_front else -self.forward_reach * 0.5 * progress
                foot[1] += self.lateral_spread * progress
            
            # Phase 0.4-0.6: Stance with compression
            elif 0.4 <= phase < 0.6:
                progress = (phase - 0.4) / 0.2
                foot[2] += self.compression_height * (1.0 - progress)
                foot[0] += self.forward_reach if is_front else -self.forward_reach * 0.5
                foot[1] += self.lateral_spread
            
            # Phase 0.6-0.8: Extension during aerial (left launch)
            elif 0.6 <= phase < 0.8:
                progress = (phase - 0.6) / 0.2
                foot[2] -= self.extension_height * progress
                foot[0] += self.forward_reach * 0.5 if is_front else -self.forward_reach * 0.3
            
            # Phase 0.8-1.0: Retracted during left landing
            else:
                progress = (phase - 0.8) / 0.2
                foot[2] += self.retraction_height * (1.0 - progress * 0.3)
                foot[0] -= self.forward_reach * 0.3 if is_front else -self.forward_reach * 0.2
        
        return foot