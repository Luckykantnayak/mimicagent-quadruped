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
        
        # Motion parameters - CALIBRATED FOR HEIGHT ENVELOPE SAFETY
        self.compression_height = 0.07  # Vertical compression distance (restored to provide margin)
        self.launch_vz = 0.50  # FURTHER REDUCED upward velocity to stay within height envelope
        self.yaw_rate_max = 3.0  # Yaw angular velocity (rad/s)
        self.spiral_extension = 0.12  # Radial extension for spiral pattern
        
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

    def smooth_step(self, t):
        """Smooth interpolation function (3rd order polynomial)"""
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on jump phase.
        Velocity profiles designed to keep base height within [0.1, 0.68] envelope.
        Uses aggressive decay to prevent excessive height accumulation.
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
            # Explosive upward velocity - smoothly ramped
            progress = (phase - 0.2) / 0.2
            vz = self.launch_vz * self.smooth_step(progress)
            # Yaw rotation begins smoothly
            yaw_rate = self.yaw_rate_max * self.smooth_step(progress)
        
        # Aerial rising [0.4, 0.6]
        elif phase < 0.6:
            # Upward velocity with aggressive cubic decay (simulating gravity)
            progress = (phase - 0.4) / 0.2
            # Using (1 - progress)^2 for more aggressive decay
            decay_factor = (1.0 - progress) * (1.0 - progress)
            vz = self.launch_vz * decay_factor
            # Constant yaw rotation
            yaw_rate = self.yaw_rate_max
        
        # Aerial peak [0.6, 0.8]
        elif phase < 0.8:
            # Transition to downward with quadratic profile
            progress = (phase - 0.6) / 0.2
            vz = -self.launch_vz * progress * progress
            # Maintain yaw rotation
            yaw_rate = self.yaw_rate_max
        
        # Descent and landing [0.8, 1.0]
        else:
            # Downward velocity decreasing smoothly toward zero
            progress = (phase - 0.8) / 0.2
            vz = -self.launch_vz * (1.0 - self.smooth_step(progress))
            # Yaw rate decreasing smoothly to zero for stable landing
            yaw_rate = self.yaw_rate_max * (1.0 - self.smooth_step(progress))
        
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
        
        # Determine leg angle offset for spiral pattern
        if leg_name == self.leg_names[0]:  # FL
            angle_offset = 0.0
        elif leg_name == self.leg_names[1]:  # FR
            angle_offset = np.pi / 2
        elif leg_name == self.leg_names[2]:  # RL
            angle_offset = 3 * np.pi / 2
        else:  # RR
            angle_offset = np.pi
        
        # Compression phase [0.0, 0.2]
        if phase < 0.2:
            # All legs compress symmetrically - foot moves up in body frame
            progress = self.smooth_step(phase / 0.2)
            foot[2] += self.compression_height * progress
        
        # Launch phase [0.2, 0.4]
        elif phase < 0.4:
            # Feet leave ground, return toward nominal position
            progress = self.smooth_step((phase - 0.2) / 0.2)
            compression_offset = self.compression_height * (1.0 - progress)
            foot[2] += compression_offset
        
        # Aerial spiral phases [0.4, 0.8]
        elif phase < 0.8:
            # Apply spiral extension with sequential timing per leg
            spiral_delay = self.spiral_phase_delays[leg_name]
            
            # Compute extension progress with individual leg delay
            if phase < 0.6:
                # Rising phase [0.4, 0.6] - legs extend
                local_phase = phase - 0.4 - spiral_delay
                local_progress = np.clip(local_phase / 0.2, 0.0, 1.0)
                extension_factor = self.smooth_step(local_progress)
            else:
                # Peak phase [0.6, 0.8] - maintain full extension
                local_phase = phase - 0.4 - spiral_delay
                if local_phase < 0.2:
                    local_progress = np.clip(local_phase / 0.2, 0.0, 1.0)
                    extension_factor = self.smooth_step(local_progress)
                else:
                    extension_factor = 1.0
            
            # Apply radial extension in body frame
            extension = self.spiral_extension * extension_factor
            foot[0] += extension * np.cos(angle_offset)
            foot[1] += extension * np.sin(angle_offset)
        
        # Descent and landing [0.8, 1.0]
        else:
            # Retract legs back to nominal stance for landing
            progress = self.smooth_step((phase - 0.8) / 0.2)
            
            # Retract from extended position
            extension = self.spiral_extension * (1.0 - progress)
            foot[0] += extension * np.cos(angle_offset)
            foot[1] += extension * np.sin(angle_offset)
        
        return foot