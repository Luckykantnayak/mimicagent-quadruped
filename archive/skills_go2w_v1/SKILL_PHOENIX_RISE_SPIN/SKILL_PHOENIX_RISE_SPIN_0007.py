from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PHOENIX_RISE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Phoenix Rise and Spin: A dramatic theatrical motion where the robot rises from 
    a fully compressed crouch to maximum standing height while executing a full 
    360-degree yaw rotation. Legs extend radially outward in a synchronized 
    'wings spreading' gesture. All four feet maintain ground contact throughout.
    
    Phase structure:
      [0.0, 0.2]: Compressed crouch initialization
      [0.2, 0.4]: Rise initiation with leg spread and yaw rotation start
      [0.4, 0.6]: Mid-rise with half leg extension and 180° rotation
      [0.6, 0.8]: Peak ascent with maximum leg extension
      [0.8, 1.0]: Peak hold with full 360° rotation complete
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Full motion cycle duration = 2 seconds
        
        # Store base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters (tuned based on diagnosis)
        self.crouch_compression = 0.4  # Less aggressive compression for smoother initial state
        self.max_radial_extension = 1.2  # Reduced from 1.5 to prevent joint limit violations
        self.rise_height = 0.24  # Reduced from 0.3 to stay within height envelope
        self.total_yaw_rotation = 2 * np.pi  # Full 360-degree rotation
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (will be set in update_base_motion)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
    def reset(self, root_pos, root_quat):
        """Reset motion generator to initial state."""
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0
        
    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent vertical velocity and yaw rate.
        
        Vertical motion:
          - [0.0, 0.2]: No vertical motion (crouch hold)
          - [0.2, 0.8]: Upward motion with smooth acceleration/deceleration
          - [0.8, 1.0]: No vertical motion (peak hold)
          
        Yaw rotation:
          - [0.0, 0.2]: No rotation (crouch hold)
          - [0.2, 1.0]: Constant yaw rate to achieve 360° by phase 1.0
        """
        
        vz = 0.0
        yaw_rate = 0.0
        
        # Vertical velocity profile with smooth blending
        if phase < 0.2:
            # Crouch initialization - no motion
            vz = 0.0
        elif phase < 0.8:
            # Rising phase: use smoothstep for smooth acceleration/deceleration
            # Map phase [0.2, 0.8] to progress [0, 1]
            rise_progress = (phase - 0.2) / (0.8 - 0.2)
            # Smoothstep function: 3t^2 - 2t^3 for smooth velocity profile
            smooth_factor = 3 * rise_progress**2 - 2 * rise_progress**3
            # Average velocity needed: rise_height / (0.6 * cycle_period)
            cycle_period = 1.0 / self.freq
            rise_duration = 0.6 * cycle_period  # Phase 0.2 to 0.8
            avg_vz = self.rise_height / rise_duration
            # Apply smoothstep scaling (derivative peaks at 1.5 * average)
            vz = avg_vz * (6 * rise_progress * (1 - rise_progress) + 1e-6)
        else:
            # Peak hold - no vertical motion
            vz = 0.0
            
        # Yaw rotation profile
        if phase < 0.2:
            # No rotation during crouch
            yaw_rate = 0.0
        else:
            # Constant yaw rate from phase 0.2 to 1.0
            # Total rotation angle: 2π radians
            # Rotation duration: 0.8 * cycle_period (phase 0.2 to 1.0)
            cycle_period = 1.0 / self.freq
            rotation_duration = 0.8 * cycle_period
            yaw_rate = self.total_yaw_rotation / rotation_duration
            
        # Set velocity commands in world frame
        self.vel_world = np.array([0.0, 0.0, vz])
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
        Compute foot position in body frame with radial extension creating wing-spread effect.
        
        All legs extend radially outward from compressed crouch to maximum extension,
        maintaining ground contact throughout. Extension is symmetric across all four legs.
        
        Foot z-coordinates remain constant to maintain ground contact.
        """
        
        # Get base foot position for this leg
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial extension factor based on phase
        if phase < 0.2:
            # Compressed crouch: feet pulled toward body center
            extension_factor = self.crouch_compression
        elif phase < 0.8:
            # Rising and extending: smooth interpolation from compressed to full extension
            # Map phase [0.2, 0.8] to extension progress [0, 1]
            extension_progress = (phase - 0.2) / (0.8 - 0.2)
            # Smoothstep for smooth extension
            smooth_progress = 3 * extension_progress**2 - 2 * extension_progress**3
            extension_factor = self.crouch_compression + (self.max_radial_extension - self.crouch_compression) * smooth_progress
        else:
            # Peak hold: maintain maximum extension
            extension_factor = self.max_radial_extension
            
        # Apply radial extension in x-y plane (body frame)
        foot_pos = base_foot.copy()
        foot_pos[0] *= extension_factor  # Scale x (forward/back)
        foot_pos[1] *= extension_factor  # Scale y (left/right)
        # Keep z constant - ground contact is maintained by IK and collision handling
        foot_pos[2] = base_foot[2]
        
        return foot_pos