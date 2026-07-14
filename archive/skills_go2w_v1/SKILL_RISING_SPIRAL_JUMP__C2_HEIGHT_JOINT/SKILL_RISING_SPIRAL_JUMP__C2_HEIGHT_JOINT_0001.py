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
        self.freq = 0.5  # Slower cycle for dramatic aerial maneuver
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.compression_depth = 0.15  # Vertical compression distance
        self.launch_velocity = 2.5  # Strong upward velocity for jump
        self.peak_height_velocity = 0.0  # Zero velocity at apex
        self.descent_velocity = -1.5  # Downward velocity during descent
        
        # Yaw rotation parameters
        self.yaw_rate_launch = 3.0  # rad/s during launch and aerial phases
        self.yaw_rate_peak = 3.0  # Maintain rotation at peak
        
        # Spiral extension parameters
        self.radial_extension = 0.25  # Outward radial extension distance
        self.spiral_height_offset = 0.1  # Slight upward offset during extension
        
        # Spiral sequence timing (sub-phase offsets within 0.4-0.6)
        self.spiral_phase_offsets = {
            leg_names[0]: 0.0,   # FL leads
            leg_names[1]: 0.25,  # FR follows
            leg_names[2]: 0.5,   # RR third
            leg_names[3]: 0.75,  # RL last
        }
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Kinematic prescription of vertical motion and yaw rotation.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.2: Compression (slight downward velocity)
        if phase < 0.2:
            progress = phase / 0.2
            vz = -0.5 * (1.0 - np.cos(np.pi * progress))  # Smooth downward
            yaw_rate = 0.0
        
        # Phase 0.2-0.4: Launch (strong upward velocity, yaw initiation)
        elif phase < 0.4:
            progress = (phase - 0.2) / 0.2
            vz = self.launch_velocity * np.sin(np.pi * 0.5 * progress)  # Ramp up
            yaw_rate = self.yaw_rate_launch * progress  # Ramp up yaw
        
        # Phase 0.4-0.6: Aerial spiral extension (velocity decreasing, yaw continues)
        elif phase < 0.6:
            progress = (phase - 0.4) / 0.2
            vz = self.launch_velocity * (1.0 - progress)  # Decelerate upward
            yaw_rate = self.yaw_rate_launch
        
        # Phase 0.6-0.8: Peak spiral (transition to descent, yaw maintained)
        elif phase < 0.8:
            progress = (phase - 0.6) / 0.2
            vz = -self.descent_velocity * progress  # Transition to downward
            yaw_rate = self.yaw_rate_peak
        
        # Phase 0.8-1.0: Descent and landing (downward velocity, yaw deceleration)
        else:
            progress = (phase - 0.8) / 0.2
            vz = self.descent_velocity
            yaw_rate = self.yaw_rate_peak * (1.0 - progress)  # Decelerate yaw to zero
        
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
        Compute foot position in body frame based on phase and leg-specific spiral timing.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg index for directional spiral
        if leg_name.startswith('FL'):
            leg_idx = 0
            angle_base = 0.0  # Forward
        elif leg_name.startswith('FR'):
            leg_idx = 1
            angle_base = -np.pi / 2  # Right
        elif leg_name.startswith('RR'):
            leg_idx = 2
            angle_base = np.pi  # Rear
        elif leg_name.startswith('RL'):
            leg_idx = 3
            angle_base = np.pi / 2  # Left
        else:
            leg_idx = 0
            angle_base = 0.0
        
        # Phase 0.0-0.2: Compression
        if phase < 0.2:
            progress = phase / 0.2
            compression = self.compression_depth * np.sin(np.pi * 0.5 * progress)
            foot[2] += compression  # Move foot upward in body frame (leg compresses)
        
        # Phase 0.2-0.4: Launch
        elif phase < 0.4:
            progress = (phase - 0.2) / 0.2
            # Foot extends downward then lifts off
            extension = self.compression_depth * (1.0 - progress)
            foot[2] += extension
        
        # Phase 0.4-0.6: Aerial spiral extension
        elif phase < 0.6:
            # Sequential spiral extension based on leg-specific offset
            spiral_progress = (phase - 0.4) / 0.2
            leg_offset = self.spiral_phase_offsets[leg_name]
            leg_spiral_phase = np.clip((spiral_progress - leg_offset) / (1.0 - leg_offset) if leg_offset < 1.0 else spiral_progress, 0.0, 1.0)
            
            # Smooth radial extension
            extension_amount = self.radial_extension * np.sin(np.pi * 0.5 * leg_spiral_phase)
            
            # Radial direction based on leg position
            radial_x = np.cos(angle_base) * extension_amount
            radial_y = np.sin(angle_base) * extension_amount
            
            foot[0] += radial_x
            foot[1] += radial_y
            foot[2] += self.spiral_height_offset * leg_spiral_phase  # Slight upward offset
        
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
            retraction_progress = 1.0 - progress  # 1.0 at start, 0.0 at end
            
            # Retract from extended position back to nominal
            radial_x = np.cos(angle_base) * self.radial_extension * retraction_progress
            radial_y = np.sin(angle_base) * self.radial_extension * retraction_progress
            
            foot[0] += radial_x
            foot[1] += radial_y
            foot[2] += self.spiral_height_offset * retraction_progress
            
            # Add slight downward preparation for landing in final portion
            if progress > 0.7:
                landing_prep = (progress - 0.7) / 0.3
                foot[2] -= 0.05 * landing_prep
        
        return foot