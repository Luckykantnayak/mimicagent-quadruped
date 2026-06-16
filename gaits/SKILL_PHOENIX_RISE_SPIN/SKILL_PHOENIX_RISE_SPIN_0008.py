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
      [0.2, 0.35]: Leg extension begins (staging phase, no upward motion)
      [0.35, 0.8]: Base rises while legs continue extending and yaw rotates
      [0.8, 1.0]: Peak hold with full 360° rotation complete
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.45  # Extended cycle duration to 2.22 seconds for smoother motion
        
        # Store base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters (tuned based on diagnosis)
        self.crouch_compression = 0.45  # Gentler compression for smoother initial state
        self.max_radial_extension = 1.15  # Further reduced to ease kinematic demands
        self.rise_height = 0.22  # Reduced to lower velocity requirements
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
          - [0.0, 0.35]: No vertical motion (crouch hold + leg extension staging)
          - [0.35, 0.8]: Upward motion with gentle constant velocity
          - [0.8, 1.0]: No vertical motion (peak hold)
          
        Yaw rotation:
          - [0.0, 0.2]: No rotation (crouch hold)
          - [0.2, 1.0]: Constant yaw rate to achieve 360° by phase 1.0
        """
        
        vz = 0.0
        yaw_rate = 0.0
        
        # Vertical velocity profile - simplified to near-constant velocity with brief ramps
        if phase < 0.35:
            # Crouch and staging phase - no vertical motion
            vz = 0.0
        elif phase < 0.8:
            # Rising phase: use gentle constant velocity with smooth entry/exit
            # Map phase [0.35, 0.8] to progress [0, 1]
            rise_progress = (phase - 0.35) / (0.8 - 0.35)
            
            # Calculate target average velocity
            cycle_period = 1.0 / self.freq
            rise_duration = 0.45 * cycle_period  # Phase 0.35 to 0.8
            avg_vz = self.rise_height / rise_duration
            
            # Apply very gentle smoothing at entry (0-10%) and exit (90-100%)
            if rise_progress < 0.1:
                # Smooth ramp-up at beginning
                ramp_factor = rise_progress / 0.1
                smooth_ramp = 3 * ramp_factor**2 - 2 * ramp_factor**3
                vz = avg_vz * smooth_ramp
            elif rise_progress > 0.9:
                # Smooth ramp-down at end
                ramp_factor = (1.0 - rise_progress) / 0.1
                smooth_ramp = 3 * ramp_factor**2 - 2 * ramp_factor**3
                vz = avg_vz * smooth_ramp
            else:
                # Constant velocity in middle section
                vz = avg_vz
        else:
            # Peak hold - no vertical motion
            vz = 0.0
            
        # Yaw rotation profile
        if phase < 0.2:
            # No rotation during crouch
            yaw_rate = 0.0
        else:
            # Constant yaw rate from phase 0.2 to 1.0
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
        maintaining ground contact throughout. Extension begins early (phase 0.2) while
        vertical rise is delayed (phase 0.35) to reduce kinematic coupling.
        """
        
        # Get base foot position for this leg
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial extension factor based on phase
        # Extension starts at 0.2, reaches full at 0.8
        if phase < 0.2:
            # Compressed crouch: feet pulled toward body center
            extension_factor = self.crouch_compression
        elif phase < 0.8:
            # Extending: smooth interpolation from compressed to full extension
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
        # Keep z constant - ground contact maintained by IK solver
        foot_pos[2] = base_foot[2]
        
        return foot_pos