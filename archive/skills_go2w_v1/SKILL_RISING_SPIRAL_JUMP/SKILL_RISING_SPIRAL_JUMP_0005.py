from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising Spiral Jump: Vertical jump with continuous yaw rotation and sequential spiral leg extension.
    
    Phase structure:
      [0.0, 0.2]: compression - all legs compress symmetrically
      [0.2, 0.4]: launch - explosive upward velocity, yaw rotation starts, feet leave ground
      [0.4, 0.6]: aerial_spiral_rising - ascending, legs extend in spiral sequence (FL→FR→RR→RL)
      [0.6, 0.8]: aerial_spiral_peak - peak altitude, legs fully extended, yaw continues
      [0.8, 1.0]: descent_and_landing - descending, legs retract, yaw slows, prepare for touchdown
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for full jump cycle
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.compression_height = 0.08  # Vertical compression distance
        self.launch_vz = 2.5  # Upward velocity during launch
        self.yaw_rate_max = 3.0  # Yaw angular velocity (rad/s)
        self.spiral_extension = 0.15  # Radial extension distance for spiral
        
        # Spiral sequence timing - staggered leg extensions
        # FL→FR→RR→RL with 0.05 phase delay between each
        self.spiral_phase_delays = {
            leg_names[0]: 0.0,   # FL leads
            leg_names[1]: 0.05,  # FR follows
            leg_names[2]: 0.10,  # RR third
            leg_names[3]: 0.15,  # RL last
        }
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on jump phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Compression phase [0.0, 0.2]
        if phase < 0.2:
            # Minimal velocity, base lowers via leg compression
            vz = 0.0
            yaw_rate = 0.0
        
        # Launch phase [0.2, 0.4]
        elif phase < 0.4:
            # Explosive upward velocity
            vz = self.launch_vz
            # Yaw rotation begins
            yaw_rate = self.yaw_rate_max
        
        # Aerial rising [0.4, 0.6]
        elif phase < 0.6:
            # Upward velocity decreasing (simulating gravity deceleration)
            progress = (phase - 0.4) / 0.2
            vz = self.launch_vz * (1.0 - progress)
            # Constant yaw rotation
            yaw_rate = self.yaw_rate_max
        
        # Aerial peak [0.6, 0.8]
        elif phase < 0.8:
            # Transition from upward to downward
            progress = (phase - 0.6) / 0.2
            vz = -self.launch_vz * progress
            # Maintain yaw rotation
            yaw_rate = self.yaw_rate_max
        
        # Descent and landing [0.8, 1.0]
        else:
            # Downward velocity decreasing toward zero
            progress = (phase - 0.8) / 0.2
            vz = -self.launch_vz * (1.0 - progress)
            # Yaw rate decreasing to zero for stable landing
            yaw_rate = self.yaw_rate_max * (1.0 - progress)
        
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame for given leg and phase.
        
        Implements:
          - Symmetric compression [0.0, 0.2]
          - Transition to nominal [0.2, 0.4]
          - Sequential spiral extension [0.4, 0.8]
          - Retraction to landing [0.8, 1.0]
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compression phase [0.0, 0.2]
        if phase < 0.2:
            # All legs compress symmetrically - foot moves up in body frame
            progress = phase / 0.2
            foot[2] += self.compression_height * progress
        
        # Launch phase [0.2, 0.4]
        elif phase < 0.4:
            # Feet leave ground, return toward nominal position
            progress = (phase - 0.2) / 0.2
            compression_offset = self.compression_height * (1.0 - progress)
            foot[2] += compression_offset
        
        # Aerial spiral phases [0.4, 0.8]
        elif phase < 0.8:
            # Apply spiral extension with sequential timing per leg
            spiral_delay = self.spiral_phase_delays[leg_name]
            
            # Determine which leg this is for directional extension
            if leg_name.startswith('FL'):
                angle_offset = 0.0
            elif leg_name.startswith('FR'):
                angle_offset = np.pi / 2
            elif leg_name.startswith('RR'):
                angle_offset = np.pi
            else:  # RL
                angle_offset = 3 * np.pi / 2
            
            # Compute extension progress with individual leg delay
            if phase < 0.6:
                # Rising phase [0.4, 0.6] - legs extend
                local_progress = (phase - 0.4 - spiral_delay) / 0.2
                local_progress = np.clip(local_progress, 0.0, 1.0)
                extension_factor = local_progress
            else:
                # Peak phase [0.6, 0.8] - maintain extension
                local_progress = (phase - 0.6 - spiral_delay) / 0.2
                local_progress = np.clip(local_progress, 0.0, 1.0)
                if local_progress < 0.5:
                    extension_factor = 1.0
                else:
                    extension_factor = 1.0
            
            # Apply radial extension in body frame
            extension = self.spiral_extension * extension_factor
            foot[0] += extension * np.cos(angle_offset)
            foot[1] += extension * np.sin(angle_offset)
        
        # Descent and landing [0.8, 1.0]
        else:
            # Retract legs back to nominal stance for landing
            progress = (phase - 0.8) / 0.2
            
            # Determine extension direction
            if leg_name.startswith('FL'):
                angle_offset = 0.0
            elif leg_name.startswith('FR'):
                angle_offset = np.pi / 2
            elif leg_name.startswith('RR'):
                angle_offset = np.pi
            else:  # RL
                angle_offset = 3 * np.pi / 2
            
            # Retract from extended position
            extension = self.spiral_extension * (1.0 - progress)
            foot[0] += extension * np.cos(angle_offset)
            foot[1] += extension * np.sin(angle_offset)
        
        return foot