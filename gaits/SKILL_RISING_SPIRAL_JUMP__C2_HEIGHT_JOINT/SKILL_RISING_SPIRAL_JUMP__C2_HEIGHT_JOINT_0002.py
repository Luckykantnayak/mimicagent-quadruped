from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising Spiral Jump: Vertical jump with continuous yaw rotation and sequential
    spiral leg extension pattern during flight.
    
    Phase breakdown:
      [0.0, 0.2]: Compression - all legs compress symmetrically
      [0.2, 0.4]: Launch - explosive upward velocity, yaw initiation, liftoff
      [0.4, 0.6]: Aerial spiral extension - legs extend radially in sequence FL→FR→RR→RL
      [0.6, 0.8]: Peak spiral - maximum height, full extension maintained
      [0.8, 1.0]: Descent and landing - leg retraction, yaw deceleration, ground contact
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5
        
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Reduced motion parameters to stay within height envelope [0.1, 0.68]
        # Nominal base height ~0.4m, so we have ~0.28m upward budget
        self.compression_depth = 0.12  # Compress to lower starting point
        self.launch_velocity = 0.65  # Reduced from 2.5 to limit peak height
        self.descent_velocity = -0.8
        
        # Yaw rotation parameters (preserved for spiral effect)
        self.yaw_rate_launch = 3.0
        self.yaw_rate_peak = 3.0
        
        # Spiral extension parameters (preserved for spiral pattern)
        self.radial_extension = 0.20  # Slightly reduced to help with joint limits
        self.spiral_height_offset = 0.08
        
        # Spiral sequence timing preserved
        self.spiral_phase_offsets = {
            leg_names[0]: 0.0,   # FL leads
            leg_names[1]: 0.25,  # FR follows
            leg_names[2]: 0.5,   # RR third
            leg_names[3]: 0.75,  # RL last
        }
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Reduced velocity magnitudes to keep base height within [0.1, 0.68]m envelope.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.2: Compression (stronger downward to lower base)
        if phase < 0.2:
            progress = phase / 0.2
            # Smooth downward velocity to compress base
            vz = -0.9 * np.sin(np.pi * progress)
            yaw_rate = 0.0
        
        # Phase 0.2-0.4: Launch (reduced upward velocity, yaw initiation)
        elif phase < 0.4:
            progress = (phase - 0.2) / 0.2
            # Smooth ramp up to peak launch velocity
            vz = self.launch_velocity * np.sin(np.pi * progress)
            yaw_rate = self.yaw_rate_launch * np.sin(np.pi * 0.5 * progress)
        
        # Phase 0.4-0.6: Aerial spiral extension (velocity decreasing, yaw continues)
        elif phase < 0.6:
            progress = (phase - 0.4) / 0.2
            # Decelerate upward velocity to zero
            vz = self.launch_velocity * np.cos(np.pi * progress)
            yaw_rate = self.yaw_rate_launch
        
        # Phase 0.6-0.8: Peak spiral (transition to descent, yaw maintained)
        elif phase < 0.8:
            progress = (phase - 0.6) / 0.2
            # Smooth transition from zero to downward velocity
            vz = self.descent_velocity * np.sin(np.pi * 0.5 * progress)
            yaw_rate = self.yaw_rate_peak
        
        # Phase 0.8-1.0: Descent and landing (downward velocity, yaw deceleration)
        else:
            progress = (phase - 0.8) / 0.2
            # Maintain downward velocity, decelerate yaw
            vz = self.descent_velocity * (1.0 - 0.3 * progress)
            yaw_rate = self.yaw_rate_peak * np.cos(np.pi * 0.5 * progress)
        
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
        Compute foot position in body frame based on phase and leg-specific spiral timing.
        Preserved spiral pattern with slightly reduced extension to help joint limits.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg direction for spiral
        if leg_name.startswith('FL'):
            angle_base = 0.0  # Forward
        elif leg_name.startswith('FR'):
            angle_base = -np.pi / 2  # Right
        elif leg_name.startswith('RR'):
            angle_base = np.pi  # Rear
        elif leg_name.startswith('RL'):
            angle_base = np.pi / 2  # Left
        else:
            angle_base = 0.0
        
        # Phase 0.0-0.2: Compression
        if phase < 0.2:
            progress = phase / 0.2
            # Smooth compression: foot moves up in body frame (leg shortens)
            compression = self.compression_depth * np.sin(np.pi * 0.5 * progress)
            foot[2] += compression
        
        # Phase 0.2-0.4: Launch
        elif phase < 0.4:
            progress = (phase - 0.2) / 0.2
            # Foot extends downward then lifts off
            compression = self.compression_depth * np.cos(np.pi * 0.5 * progress)
            foot[2] += compression
        
        # Phase 0.4-0.6: Aerial spiral extension
        elif phase < 0.6:
            # Sequential spiral extension based on leg-specific offset
            spiral_progress = (phase - 0.4) / 0.2
            leg_offset = self.spiral_phase_offsets[leg_name]
            
            # Calculate leg-specific spiral phase with smooth onset
            if leg_offset < 1.0:
                leg_spiral_phase = np.clip((spiral_progress - leg_offset) / (1.0 - leg_offset), 0.0, 1.0)
            else:
                leg_spiral_phase = spiral_progress
            
            # Smooth radial extension with easing
            extension_amount = self.radial_extension * (1.0 - np.cos(np.pi * leg_spiral_phase)) * 0.5
            
            # Radial direction based on leg position
            radial_x = np.cos(angle_base) * extension_amount
            radial_y = np.sin(angle_base) * extension_amount
            
            foot[0] += radial_x
            foot[1] += radial_y
            foot[2] += self.spiral_height_offset * (1.0 - np.cos(np.pi * leg_spiral_phase)) * 0.5
        
        # Phase 0.6-0.8: Peak spiral (maintain full extension)
        elif phase < 0.8:
            # Full radial extension maintained
            radial_x = np.cos(angle_base) * self.radial_extension
            radial_y = np.sin(angle_base) * self.radial_extension
            
            foot[0] += radial_x
            foot[1] += radial_y
            foot[2] += self.spiral_height_offset
        
        # Phase 0.8-1.0: Descent and landing (retract legs)
        else:
            progress = (phase - 0.8) / 0.2
            # Smooth retraction from extended position back to nominal
            retraction_progress = np.cos(np.pi * 0.5 * progress)
            
            radial_x = np.cos(angle_base) * self.radial_extension * retraction_progress
            radial_y = np.sin(angle_base) * self.radial_extension * retraction_progress
            
            foot[0] += radial_x
            foot[1] += radial_y
            foot[2] += self.spiral_height_offset * retraction_progress
            
            # Slight downward preparation for landing
            if progress > 0.6:
                landing_prep = (progress - 0.6) / 0.4
                foot[2] -= 0.04 * (1.0 - np.cos(np.pi * 0.5 * landing_prep))
        
        return foot