from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PHOENIX_RISE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Phoenix Rise and Spin: A dramatic theatrical motion where the robot rises from 
    a fully compressed crouch to maximum standing height while executing a full 
    360-degree yaw rotation. Legs extend radially outward in a synchronized 
    'wings spreading' gesture. All four feet maintain ground contact throughout.
    
    Phase structure (SEQUENTIAL to prevent rotation-induced foot motion):
      [0.0, 0.2]: Compressed crouch initialization
      [0.2, 0.6]: Legs extend radially and base rises (NO rotation)
      [0.6, 0.95]: Base rotates 360° while holding extended pose
      [0.95, 1.0]: Final hold
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Full motion cycle duration = 2 seconds
        
        # Store base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - very conservative to ensure contact maintenance
        self.crouch_compression = 0.55  # Gentle compression
        self.max_radial_extension = 1.08  # Minimal extension to reduce kinematic stress
        self.rise_height = 0.18  # Conservative rise height
        self.total_yaw_rotation = 2 * np.pi  # Full 360-degree rotation
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
    def reset(self, root_pos, root_quat):
        """Reset motion generator to initial state."""
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0
        
    def update_base_motion(self, phase, dt):
        """
        Update base using SEQUENTIAL phasing to avoid rotation during extension.
        
        Vertical motion:
          - [0.0, 0.2]: No motion (crouch hold)
          - [0.2, 0.6]: Upward motion (extension phase, no rotation)
          - [0.6, 1.0]: No vertical motion (rotation and hold phases)
          
        Yaw rotation:
          - [0.0, 0.6]: No rotation (crouch and extension phases)
          - [0.6, 0.95]: Rotation executes while legs are fully extended
          - [0.95, 1.0]: Hold final orientation
        """
        
        vz = 0.0
        yaw_rate = 0.0
        
        # Vertical velocity profile - only during extension phase
        if phase < 0.2:
            vz = 0.0
        elif phase < 0.6:
            # Rising phase with smooth velocity profile
            rise_progress = (phase - 0.2) / (0.6 - 0.2)
            
            cycle_period = 1.0 / self.freq
            rise_duration = 0.4 * cycle_period
            avg_vz = self.rise_height / rise_duration
            
            # Gentle ramps at entry and exit, constant in middle
            if rise_progress < 0.15:
                ramp = rise_progress / 0.15
                smooth_ramp = 3 * ramp**2 - 2 * ramp**3
                vz = avg_vz * smooth_ramp
            elif rise_progress > 0.85:
                ramp = (1.0 - rise_progress) / 0.15
                smooth_ramp = 3 * ramp**2 - 2 * ramp**3
                vz = avg_vz * smooth_ramp
            else:
                vz = avg_vz
        else:
            vz = 0.0
            
        # Yaw rotation profile - ONLY after extension completes
        if phase < 0.6:
            # No rotation during crouch and extension
            yaw_rate = 0.0
        elif phase < 0.95:
            # Rotation phase - constant rate
            cycle_period = 1.0 / self.freq
            rotation_duration = 0.35 * cycle_period
            yaw_rate = self.total_yaw_rotation / rotation_duration
        else:
            # Final hold - no rotation
            yaw_rate = 0.0
            
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
        Compute foot position in body frame with radial extension.
        
        CRITICAL FIX: Scale z-coordinate along with x and y to maintain proportional
        leg geometry during extension. This prevents the kinematic solver from being
        asked to reach positions that require excessive leg length.
        """
        
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial extension factor
        if phase < 0.2:
            # Compressed crouch
            extension_factor = self.crouch_compression
        elif phase < 0.6:
            # Extension phase - completes before rotation starts
            extension_progress = (phase - 0.2) / (0.6 - 0.2)
            smooth_progress = 3 * extension_progress**2 - 2 * extension_progress**3
            extension_factor = self.crouch_compression + \
                             (self.max_radial_extension - self.crouch_compression) * smooth_progress
        else:
            # Hold extended pose during rotation and final phases
            extension_factor = self.max_radial_extension
            
        # Apply radial extension to ALL coordinates to maintain proportional leg geometry
        # This is critical: scaling only x-y but not z increases required leg length
        # and creates unreachable foot positions
        foot_pos = base_foot.copy()
        foot_pos[0] *= extension_factor
        foot_pos[1] *= extension_factor
        foot_pos[2] *= extension_factor  # Scale z proportionally
        
        return foot_pos