from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PHOENIX_RISE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Phoenix Rise and Spin motion generator.
    
    Robot starts in deep crouch, rises to maximum height while rotating 360 degrees.
    Legs extend radially outward and upward in body frame, creating a wing-spreading effect.
    All four feet maintain ground contact throughout the motion.
    
    Phase structure:
      [0.0, 0.2]: Initial crouch - stationary, legs compressed
      [0.2, 0.4]: Rise initiation - upward velocity and yaw rotation begin
      [0.4, 0.6]: Mid-rise spread - continued rise and rotation, legs at mid-extension
      [0.6, 0.8]: Peak ascent - approaching max height, continued rotation
      [0.8, 1.0]: Peak hold and completion - max height maintained, 360° rotation completes
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Store initial foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.crouch_height_reduction = 0.15  # How much to compress legs in crouch (meters)
        self.max_extension_factor = 1.8  # Radial extension multiplier at peak
        self.upward_component_max = 0.12  # Maximum upward reach in body frame (meters)
        
        # Base motion parameters
        self.rise_height = 0.25  # Total vertical rise (meters)
        self.upward_velocity = 0.3  # Constant upward velocity during rise (m/s)
        self.total_rotation = 2 * np.pi  # 360 degrees in radians
        self.rotation_duration = 0.8  # Duration over which rotation occurs (phase 0.2 to 1.0)
        self.yaw_rate = self.total_rotation / (self.rotation_duration / self.freq)  # rad/s
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Pre-compute initial radial distances and angles for each leg
        self.leg_radial_info = {}
        for leg in self.leg_names:
            pos = self.base_feet_pos_body[leg]
            radius = np.sqrt(pos[0]**2 + pos[1]**2)
            angle = np.arctan2(pos[1], pos[0])
            self.leg_radial_info[leg] = {
                'base_radius': radius,
                'angle': angle,
                'base_z': pos[2]
            }

    def update_base_motion(self, phase, dt):
        """
        Update base pose using velocity commands based on current phase.
        
        Phase [0.0, 0.2]: Stationary
        Phase [0.2, 0.8]: Constant upward velocity and yaw rate
        Phase [0.8, 1.0]: Decelerate upward velocity to maintain height, continue yaw
        """
        
        if phase < 0.2:
            # Initial crouch - no motion
            vz = 0.0
            yaw_rate = 0.0
        elif phase < 0.8:
            # Rising and rotating
            vz = self.upward_velocity
            yaw_rate = self.yaw_rate
        else:
            # Peak hold - decelerate vertical motion, continue rotation
            # Smoothly reduce vertical velocity to zero
            decel_progress = (phase - 0.8) / 0.2
            vz = self.upward_velocity * (1.0 - decel_progress)
            yaw_rate = self.yaw_rate
        
        # Set velocity commands
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
        Compute foot position in body frame for given leg and phase.
        
        Feet extend radially outward and upward as phase progresses.
        Maintains radial symmetry around body center.
        """
        
        # Get base position and radial info
        base_pos = self.base_feet_pos_body[leg_name]
        radial_info = self.leg_radial_info[leg_name]
        base_radius = radial_info['base_radius']
        angle = radial_info['angle']
        base_z = radial_info['base_z']
        
        # Compute extension factor based on phase
        if phase < 0.2:
            # Crouch phase - legs compressed
            radial_factor = 0.6  # Compress toward body center
            z_offset = -self.crouch_height_reduction
            upward_reach = 0.0
        elif phase < 0.4:
            # Rise initiation
            sub_phase = (phase - 0.2) / 0.2
            radial_factor = 0.6 + 0.4 * sub_phase  # 0.6 -> 1.0
            z_offset = -self.crouch_height_reduction * (1.0 - sub_phase)
            upward_reach = self.upward_component_max * 0.2 * sub_phase
        elif phase < 0.6:
            # Mid-rise spread
            sub_phase = (phase - 0.4) / 0.2
            radial_factor = 1.0 + 0.4 * sub_phase  # 1.0 -> 1.4
            z_offset = 0.0
            upward_reach = self.upward_component_max * (0.2 + 0.3 * sub_phase)
        elif phase < 0.8:
            # Peak ascent
            sub_phase = (phase - 0.6) / 0.2
            radial_factor = 1.4 + 0.3 * sub_phase  # 1.4 -> 1.7
            z_offset = 0.0
            upward_reach = self.upward_component_max * (0.5 + 0.4 * sub_phase)
        else:
            # Peak hold and completion
            sub_phase = (phase - 0.8) / 0.2
            radial_factor = 1.7 + 0.1 * sub_phase  # 1.7 -> 1.8
            z_offset = 0.0
            upward_reach = self.upward_component_max * (0.9 + 0.1 * sub_phase)
        
        # Compute new foot position in body frame
        new_radius = base_radius * radial_factor
        foot_x = new_radius * np.cos(angle)
        foot_y = new_radius * np.sin(angle)
        foot_z = base_z + z_offset + upward_reach
        
        return np.array([foot_x, foot_y, foot_z])