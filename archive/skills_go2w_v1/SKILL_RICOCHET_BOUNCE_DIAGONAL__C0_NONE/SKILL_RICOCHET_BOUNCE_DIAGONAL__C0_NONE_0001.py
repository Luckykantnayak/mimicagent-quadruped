from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal ricochet bounce skill with alternating left/right yaw-biased explosive bounces.
    
    Motion pattern:
    - [0.0-0.2]: Left diagonal compression (FL+RL), yaw left
    - [0.2-0.4]: Explosive aerial launch diagonally forward-right
    - [0.4-0.6]: Right diagonal landing/compression (FR+RR), yaw right
    - [0.6-0.8]: Explosive aerial launch diagonally forward-left
    - [0.8-1.0]: Left diagonal landing (FL+RL), yaw neutralization
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.2
        
        # Store base foot positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.compression_depth = 0.12
        self.extension_height = 0.15
        self.lateral_offset = 0.08
        self.rearward_shift = 0.06
        
        # Velocity parameters
        self.forward_velocity_cruise = 0.5
        self.forward_velocity_launch = 2.5
        self.lateral_velocity_launch = 0.8
        self.vertical_velocity_launch = 1.8
        self.vertical_velocity_landing = -1.2
        
        # Yaw parameters (radians)
        self.yaw_rate_compression = 4.5  # ~30 degrees over 0.2 phase duration
        self.yaw_rate_neutralize = 3.0
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Implements compression, launch, aerial, and landing dynamics.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        if 0.0 <= phase < 0.2:
            # Left compression and yaw
            progress = phase / 0.2
            vx = self.forward_velocity_cruise * (1.0 - 0.5 * progress)
            vy = 0.0
            vz = -self.compression_depth * 5.0 * (1.0 - np.cos(np.pi * progress))
            yaw_rate = self.yaw_rate_compression
            
        elif 0.2 <= phase < 0.4:
            # Left launch aerial (forward-right diagonal)
            progress = (phase - 0.2) / 0.2
            vx = self.forward_velocity_launch * np.exp(-2.0 * progress)
            vy = self.lateral_velocity_launch * (1.0 - progress)
            # Parabolic trajectory: launch up, then fall
            vz = self.vertical_velocity_launch * (1.0 - 2.0 * progress)
            yaw_rate = 0.0
            
        elif 0.4 <= phase < 0.6:
            # Right landing compression and yaw
            progress = (phase - 0.4) / 0.2
            vx = self.forward_velocity_cruise * (0.5 + 0.5 * progress)
            vy = self.lateral_velocity_launch * (1.0 - progress) * 0.3
            vz = self.vertical_velocity_landing * progress
            yaw_rate = -self.yaw_rate_compression
            
        elif 0.6 <= phase < 0.8:
            # Right launch aerial (forward-left diagonal)
            progress = (phase - 0.6) / 0.2
            vx = self.forward_velocity_launch * np.exp(-2.0 * progress)
            vy = -self.lateral_velocity_launch * (1.0 - progress)
            # Parabolic trajectory
            vz = self.vertical_velocity_launch * (1.0 - 2.0 * progress)
            yaw_rate = 0.0
            
        else:  # 0.8 <= phase < 1.0
            # Left landing and yaw neutralization
            progress = (phase - 0.8) / 0.2
            vx = self.forward_velocity_cruise * (0.5 + 0.5 * progress)
            vy = -self.lateral_velocity_launch * (1.0 - progress) * 0.3
            vz = self.vertical_velocity_landing * progress
            yaw_rate = self.yaw_rate_neutralize
        
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
        Compute foot position in body frame based on phase and leg.
        Diagonal pairs (FL+RL, FR+RR) alternate between stance and swing.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front_leg = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        if is_left_leg:
            # FL and RL: stance in [0.0-0.2] and [0.8-1.0], swing otherwise
            if 0.0 <= phase < 0.2:
                # Compression stance - left diagonal
                progress = phase / 0.2
                foot[2] += self.compression_depth * progress
                foot[0] -= self.rearward_shift * progress
                
            elif 0.2 <= phase < 0.4:
                # Aerial extension after left launch
                progress = (phase - 0.2) / 0.2
                foot[2] -= self.extension_height * np.sin(np.pi * progress)
                foot[0] += self.rearward_shift * (1.0 - progress)
                
            elif 0.4 <= phase < 0.6:
                # Swing retraction while right diagonal in stance
                progress = (phase - 0.4) / 0.2
                foot[2] -= self.extension_height * (1.0 - progress) * 0.5
                foot[1] -= self.lateral_offset * np.sin(np.pi * progress) * 0.3
                
            elif 0.6 <= phase < 0.8:
                # Aerial extension during right launch
                progress = (phase - 0.6) / 0.2
                foot[2] -= self.extension_height * np.sin(np.pi * progress)
                foot[0] += self.rearward_shift * progress * 0.5
                
            else:  # 0.8 <= phase < 1.0
                # Landing and compression - left diagonal
                progress = (phase - 0.8) / 0.2
                foot[2] += self.compression_depth * progress
                foot[0] -= self.rearward_shift * progress * 0.5
                
        else:
            # FR and RR: stance in [0.4-0.6], swing otherwise
            if 0.0 <= phase < 0.2:
                # Swing retraction while left diagonal in stance
                progress = phase / 0.2
                foot[2] -= self.extension_height * (1.0 - progress) * 0.5
                foot[1] += self.lateral_offset * np.sin(np.pi * progress) * 0.3
                
            elif 0.2 <= phase < 0.4:
                # Aerial extension, preparing for right landing
                progress = (phase - 0.2) / 0.2
                foot[2] -= self.extension_height * np.sin(np.pi * progress)
                foot[0] += self.rearward_shift * progress
                
            elif 0.4 <= phase < 0.6:
                # Compression stance - right diagonal
                progress = (phase - 0.4) / 0.2
                foot[2] += self.compression_depth * progress
                foot[0] -= self.rearward_shift * progress
                
            elif 0.6 <= phase < 0.8:
                # Aerial extension after right launch
                progress = (phase - 0.6) / 0.2
                foot[2] -= self.extension_height * np.sin(np.pi * progress)
                foot[0] += self.rearward_shift * (1.0 - progress)
                
            else:  # 0.8 <= phase < 1.0
                # Swing retraction while left diagonal lands
                progress = (phase - 0.8) / 0.2
                foot[2] -= self.extension_height * (1.0 - progress) * 0.5
                foot[1] += self.lateral_offset * np.sin(np.pi * progress) * 0.3
        
        return foot