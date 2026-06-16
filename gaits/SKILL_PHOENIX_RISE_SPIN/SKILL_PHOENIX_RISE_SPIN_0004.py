from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PHOENIX_RISE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Phoenix Rise and Spin motion generator.
    
    Robot starts in deep crouch, rises to maximum height while rotating 360 degrees.
    Legs extend radially outward in body frame, creating a wing-spreading effect.
    All four feet maintain ground contact throughout the motion.
    
    Phase structure:
      [0.0, 0.2]: Initial crouch - base lowered, legs compressed radially
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
        
        # Motion parameters - tuned to avoid joint limits and maintain ground contact
        self.max_extension_factor = 1.08  # Radial extension multiplier at peak (conservative for joint safety)
        
        # Base motion parameters
        self.crouch_depth = 0.05  # Base descends during crouch (meters, reduced for safety)
        self.rise_height = 0.15  # Total vertical rise from crouch (meters, reduced for kinematic feasibility)
        self.total_rotation = 2 * np.pi  # 360 degrees in radians
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Initialize base height from foot positions
        avg_foot_z = np.mean([pos[2] for pos in self.base_feet_pos_body.values()])
        self.initial_base_height = -avg_foot_z
        if self.initial_base_height < 0.2:
            self.initial_base_height = 0.28
        
        # Ground level in world frame
        self.ground_level = 0.0
        
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
        Update base pose based on current phase.
        
        Phase [0.0, 0.2]: Crouch - base descends
        Phase [0.2, 0.8]: Rising and rotating - base ascends with yaw
        Phase [0.8, 1.0]: Peak hold - maintain height, complete rotation
        """
        
        # Target base height based on phase with smooth transitions
        if phase < 0.2:
            # Crouch phase: smoothly descend to crouch depth
            crouch_progress = phase / 0.2
            crouch_smoothed = 0.5 * (1.0 - np.cos(crouch_progress * np.pi))
            target_z = self.initial_base_height - self.crouch_depth * crouch_smoothed
        elif phase < 0.8:
            # Rise phase: smoothly ascend from crouch to peak
            rise_progress = (phase - 0.2) / 0.6
            rise_smoothed = 0.5 * (1.0 - np.cos(rise_progress * np.pi))
            target_z = (self.initial_base_height - self.crouch_depth) + self.rise_height * rise_smoothed
        else:
            # Peak hold: maintain maximum height
            target_z = self.initial_base_height - self.crouch_depth + self.rise_height
        
        # Target yaw based on phase with smooth acceleration and deceleration
        if phase < 0.2:
            target_yaw = 0.0
        else:
            # Rotation from phase 0.2 to 1.0
            rotation_progress = (phase - 0.2) / 0.8
            rotation_smoothed = 0.5 * (1.0 - np.cos(rotation_progress * np.pi))
            target_yaw = self.total_rotation * rotation_smoothed
        
        # Compute velocities based on target derivatives
        if dt > 0:
            vz = (target_z - self.root_pos[2]) / dt
            
            current_yaw = np.arctan2(2.0 * (self.root_quat[0] * self.root_quat[3] + self.root_quat[1] * self.root_quat[2]),
                                     1.0 - 2.0 * (self.root_quat[2]**2 + self.root_quat[3]**2))
            yaw_diff = target_yaw - current_yaw
            while yaw_diff > np.pi:
                yaw_diff -= 2 * np.pi
            while yaw_diff < -np.pi:
                yaw_diff += 2 * np.pi
            yaw_rate = yaw_diff / dt
        else:
            vz = 0.0
            yaw_rate = 0.0
        
        self.vel_world = np.array([0.0, 0.0, vz])
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
        Compute foot position in body frame for given leg and phase.
        
        Feet extend radially outward as phase progresses.
        Z-coordinate adjusted to maintain ground contact as base rises.
        """
        
        radial_info = self.leg_radial_info[leg_name]
        base_radius = radial_info['base_radius']
        angle = radial_info['angle']
        base_z = radial_info['base_z']
        
        # Compute radial extension factor based on phase with smooth progression
        if phase < 0.2:
            # Crouch phase - legs compressed radially toward body
            radial_factor = 0.88
        elif phase < 0.4:
            # Rise initiation - begin extending
            sub_phase = (phase - 0.2) / 0.2
            sub_smoothed = 0.5 * (1.0 - np.cos(sub_phase * np.pi))
            radial_factor = 0.88 + 0.08 * sub_smoothed  # 0.88 -> 0.96
        elif phase < 0.6:
            # Mid-rise spread
            sub_phase = (phase - 0.4) / 0.2
            sub_smoothed = 0.5 * (1.0 - np.cos(sub_phase * np.pi))
            radial_factor = 0.96 + 0.06 * sub_smoothed  # 0.96 -> 1.02
        elif phase < 0.8:
            # Peak ascent
            sub_phase = (phase - 0.6) / 0.2
            sub_smoothed = 0.5 * (1.0 - np.cos(sub_phase * np.pi))
            radial_factor = 1.02 + 0.04 * sub_smoothed  # 1.02 -> 1.06
        else:
            # Peak hold and completion
            sub_phase = (phase - 0.8) / 0.2
            sub_smoothed = 0.5 * (1.0 - np.cos(sub_phase * np.pi))
            radial_factor = 1.06 + 0.02 * sub_smoothed  # 1.06 -> 1.08
        
        # Compute new foot position in body frame (radial extension in X-Y plane)
        new_radius = base_radius * radial_factor
        foot_x = new_radius * np.cos(angle)
        foot_y = new_radius * np.sin(angle)
        
        # Compute current base height to maintain ground contact
        if phase < 0.2:
            crouch_progress = phase / 0.2
            crouch_smoothed = 0.5 * (1.0 - np.cos(crouch_progress * np.pi))
            current_base_height = self.initial_base_height - self.crouch_depth * crouch_smoothed
        elif phase < 0.8:
            rise_progress = (phase - 0.2) / 0.6
            rise_smoothed = 0.5 * (1.0 - np.cos(rise_progress * np.pi))
            current_base_height = (self.initial_base_height - self.crouch_depth) + self.rise_height * rise_smoothed
        else:
            current_base_height = self.initial_base_height - self.crouch_depth + self.rise_height
        
        # Correct approach: as base rises, foot must extend further down in body frame to maintain ground contact
        # foot_z in body frame = (ground_level - current_base_height) + leg_specific_offset
        # leg_specific_offset preserves initial foot z variation between legs
        base_height_change = current_base_height - self.initial_base_height
        foot_z = base_z - base_height_change
        
        return np.array([foot_x, foot_y, foot_z])