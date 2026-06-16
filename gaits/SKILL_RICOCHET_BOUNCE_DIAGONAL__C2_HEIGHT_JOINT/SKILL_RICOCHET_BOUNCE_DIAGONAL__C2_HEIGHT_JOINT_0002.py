from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Ricochet bounce diagonal gait: alternating left/right diagonal bounces with yaw rotation.
    
    Phase structure:
      [0.0, 0.2]: Left compression + negative yaw
      [0.2, 0.4]: Right diagonal launch (aerial)
      [0.4, 0.6]: Right compression + positive yaw
      [0.6, 0.8]: Left diagonal launch (aerial)
      [0.8, 1.0]: Landing transition on left side
    
    Contact pattern:
      - FL+RL in contact during [0.0-0.2] and [0.8-1.0]
      - FR+RR in contact during [0.4-0.6]
      - All feet airborne during [0.2-0.4] and [0.6-0.8]
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.7  # Reduced from 0.8 to allow smoother motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - reduced to prevent joint limit violations
        self.compression_depth = 0.04  # Reduced from 0.08 to prevent over-compression
        self.launch_height = 0.05  # Reduced from 0.12 to prevent over-extension
        self.step_forward = 0.10  # Reduced from 0.15 to keep feet closer
        self.lateral_displacement = 0.06  # Reduced from 0.08 to limit lateral reach
        
        # Base velocity parameters - reduced to stay within height envelope
        self.vx_forward = 0.5  # Forward velocity (m/s)
        self.vy_lateral_amp = 0.6  # Reduced from 0.8 for smoother lateral motion
        self.vz_compression = -0.3  # Reduced from -0.4 for gentler compression
        self.vz_launch = 0.75  # Reduced from 1.2 to prevent exceeding height envelope
        self.yaw_rate_amp = 1.5  # Reduced from 2.0 for smoother yaw transitions
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base velocities based on phase to create ricochet bounce pattern.
        Reduced amplitudes to stay within height envelope and prevent joint violations.
        """
        vx = self.vx_forward
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Smooth transition helper
        def smooth_step(x):
            return x * x * (3.0 - 2.0 * x)
        
        # Phase [0.0, 0.2]: Left compression + negative yaw
        if phase < 0.2:
            local_phase = phase / 0.2
            smooth_phase = smooth_step(local_phase)
            vz = self.vz_compression * np.sin(np.pi * local_phase) * 0.7
            yaw_rate = -self.yaw_rate_amp * np.sin(np.pi * local_phase)
            
        # Phase [0.2, 0.4]: Right diagonal launch (aerial)
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            smooth_phase = smooth_step(local_phase)
            vz = self.vz_launch * np.sin(np.pi * local_phase)
            vy = self.vy_lateral_amp * smooth_phase
            yaw_rate = -self.yaw_rate_amp * (1.0 - local_phase) * 0.3
            
        # Phase [0.4, 0.6]: Right compression + positive yaw
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            smooth_phase = smooth_step(local_phase)
            vz = self.vz_compression * np.sin(np.pi * local_phase) * 0.7
            yaw_rate = self.yaw_rate_amp * np.sin(np.pi * local_phase)
            vy = self.vy_lateral_amp * (1.0 - smooth_phase) * 0.3
            
        # Phase [0.6, 0.8]: Left diagonal launch (aerial)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            smooth_phase = smooth_step(local_phase)
            vz = self.vz_launch * np.sin(np.pi * local_phase)
            vy = -self.vy_lateral_amp * smooth_phase
            yaw_rate = self.yaw_rate_amp * (1.0 - local_phase) * 0.3
            
        # Phase [0.8, 1.0]: Landing transition on left side
        else:
            local_phase = (phase - 0.8) / 0.2
            smooth_phase = smooth_step(local_phase)
            vz = self.vz_compression * np.sin(np.pi * local_phase) * 0.5
            yaw_rate = -self.yaw_rate_amp * smooth_phase * 0.2
            vy = -self.vy_lateral_amp * (1.0 - smooth_phase) * 0.2
        
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
        Reduced vertical displacements to prevent joint limit violations.
        
        Contact schedule:
          FL, RL: stance [0.0-0.2], swing [0.2-0.8], stance [0.8-1.0]
          FR, RR: swing [0.0-0.4], stance [0.4-0.6], swing [0.6-1.0]
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left = leg_name in ['FL', 'RL']
        is_front = leg_name in ['FL', 'FR']
        
        # Smooth transition helper
        def smooth_step(x):
            return x * x * (3.0 - 2.0 * x)
        
        if is_left:
            # FL, RL: Left-side legs
            if phase < 0.2:
                # Stance: compression - minimal vertical motion since base is also compressing
                local_phase = phase / 0.2
                foot[2] -= self.compression_depth * np.sin(np.pi * local_phase) * 0.5
                
            elif phase < 0.4:
                # Swing: lift during right launch - reduced lift since base is rising
                local_phase = (phase - 0.2) / 0.2
                smooth_phase = smooth_step(local_phase)
                foot[0] += self.step_forward * smooth_phase * 0.4
                foot[1] -= self.lateral_displacement * smooth_phase * 0.7
                foot[2] += self.launch_height * np.sin(np.pi * local_phase) * 0.6
                
            elif phase < 0.6:
                # Swing: continue forward motion with minimal vertical
                local_phase = (phase - 0.4) / 0.2
                smooth_phase = smooth_step(local_phase)
                foot[0] += self.step_forward * (0.4 + smooth_phase * 0.3)
                foot[1] -= self.lateral_displacement * (0.7 - smooth_phase * 0.3)
                foot[2] += self.launch_height * 0.2
                
            elif phase < 0.8:
                # Swing: prepare for landing during left launch
                local_phase = (phase - 0.6) / 0.2
                smooth_phase = smooth_step(local_phase)
                foot[0] += self.step_forward * (0.7 + smooth_phase * 0.3)
                foot[1] -= self.lateral_displacement * (0.4 - smooth_phase * 0.4)
                foot[2] += self.launch_height * np.sin(np.pi * local_phase) * 0.4
                
            else:
                # Stance: landing and compression
                local_phase = (phase - 0.8) / 0.2
                smooth_phase = smooth_step(local_phase)
                foot[0] += self.step_forward
                foot[2] -= self.compression_depth * np.sin(np.pi * local_phase) * 0.3
                
        else:
            # FR, RR: Right-side legs
            if phase < 0.2:
                # Swing: prepare for right launch
                local_phase = phase / 0.2
                smooth_phase = smooth_step(local_phase)
                foot[0] += self.step_forward * smooth_phase * 0.2
                foot[1] += self.lateral_displacement * smooth_phase * 0.4
                foot[2] += self.launch_height * smooth_phase * 0.3
                
            elif phase < 0.4:
                # Swing: extend during right launch - reduced since base is rising
                local_phase = (phase - 0.2) / 0.2
                smooth_phase = smooth_step(local_phase)
                foot[0] += self.step_forward * (0.2 + smooth_phase * 0.3)
                foot[1] += self.lateral_displacement * (0.4 + smooth_phase * 0.4)
                foot[2] += self.launch_height * (0.3 + np.sin(np.pi * local_phase) * 0.3)
                
            elif phase < 0.6:
                # Stance: landing and compression on right side
                local_phase = (phase - 0.4) / 0.2
                foot[0] += self.step_forward * 0.5
                foot[1] += self.lateral_displacement * 0.8
                foot[2] -= self.compression_depth * np.sin(np.pi * local_phase) * 0.5
                
            elif phase < 0.8:
                # Swing: lift during left launch - reduced since base is rising
                local_phase = (phase - 0.6) / 0.2
                smooth_phase = smooth_step(local_phase)
                foot[0] += self.step_forward * (0.5 + smooth_phase * 0.4)
                foot[1] += self.lateral_displacement * (0.8 - smooth_phase * 0.6)
                foot[2] += self.launch_height * np.sin(np.pi * local_phase) * 0.6
                
            else:
                # Swing: continue motion
                local_phase = (phase - 0.8) / 0.2
                smooth_phase = smooth_step(local_phase)
                foot[0] += self.step_forward * (0.9 + smooth_phase * 0.1)
                foot[1] += self.lateral_displacement * (0.2 - smooth_phase * 0.2)
                foot[2] += self.launch_height * (1.0 - smooth_phase) * 0.2
        
        return foot