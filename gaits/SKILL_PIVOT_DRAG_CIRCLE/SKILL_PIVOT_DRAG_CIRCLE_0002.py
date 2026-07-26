from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PIVOT_DRAG_CIRCLE_MotionGenerator(BaseMotionGenerator):
    """
    Robot performs a full 360-degree circular motion by sequentially pivoting 
    around three different fixed legs (RR → RL → FR) while the other three legs 
    drag along the ground in coordinated arcs.
    
    Phase structure:
      [0.0, 0.33]: Pivot around RR, FL/FR/RL drag
      [0.33, 0.67]: Pivot around RL, FL/FR/RR drag
      [0.67, 1.0]: Pivot around FR, FL/RL/RR drag
    
    All legs maintain continuous ground contact throughout.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.3  # Slow motion for stability during dragging

        # Store initial foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Identify legs by name
        self.FL = [l for l in leg_names if l.startswith('FL')][0]
        self.FR = [l for l in leg_names if l.startswith('FR')][0]
        self.RL = [l for l in leg_names if l.startswith('RL')][0]
        self.RR = [l for l in leg_names if l.startswith('RR')][0]

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Motion parameters
        self.total_yaw_rotation = 2 * np.pi  # 360 degrees total
        self.yaw_per_segment = self.total_yaw_rotation / 3.0  # 120 degrees per segment
        self.segment_duration = 1.0 / (3.0 * self.freq)
        
        # Reduced motion radius to stay within workspace
        self.motion_radius_scale = 0.25  # Scale factor for circular motion
        
        # Angular velocity for constant yaw rate per segment
        self.yaw_rate = self.yaw_per_segment / self.segment_duration

        # Store world positions for all legs at segment start
        self.segment_start_world_positions = {}
        self.segment_start_base_pose = {}

    def reset(self, root_pos, root_quat):
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0
        self.segment_start_world_positions = {}
        self.segment_start_base_pose = {}

    def get_current_pivot_leg(self, phase):
        """Returns the name of the current pivot leg based on phase."""
        if phase < 0.33:
            return self.RR
        elif phase < 0.67:
            return self.RL
        else:
            return self.FR

    def get_segment_phase(self, phase):
        """Returns normalized phase within current segment [0, 1]."""
        if phase < 0.33:
            return phase / 0.33
        elif phase < 0.67:
            return (phase - 0.33) / 0.34
        else:
            return (phase - 0.67) / 0.33

    def get_segment_index(self, phase):
        """Returns segment index: 0, 1, or 2."""
        if phase < 0.33:
            return 0
        elif phase < 0.67:
            return 1
        else:
            return 2

    def update_base_motion(self, phase, dt):
        """
        Update base motion to create circular rotation around current pivot leg.
        Each segment rotates ~120 degrees around a different pivot point.
        """
        pivot_leg = self.get_current_pivot_leg(phase)
        segment_phase = self.get_segment_phase(phase)
        segment_idx = self.get_segment_index(phase)
        
        # Store segment start configuration
        segment_key = f"{segment_idx}"
        if segment_key not in self.segment_start_world_positions:
            self.segment_start_world_positions[segment_key] = {}
            for leg_name in self.leg_names:
                foot_body = self.base_feet_pos_body[leg_name].copy()
                foot_world = body_to_world_position(foot_body, self.root_pos, self.root_quat)
                self.segment_start_world_positions[segment_key][leg_name] = foot_world.copy()
            self.segment_start_base_pose[segment_key] = {
                'pos': self.root_pos.copy(),
                'quat': self.root_quat.copy()
            }
        
        # Get pivot leg position in world frame (fixed throughout segment)
        pivot_world = self.segment_start_world_positions[segment_key][pivot_leg].copy()
        
        # Get initial base position at segment start
        base_start_pos = self.segment_start_base_pose[segment_key]['pos']
        base_start_quat = self.segment_start_base_pose[segment_key]['quat']
        
        # Vector from pivot to base center at segment start (in world frame, xy only)
        r_pivot_to_base_start = base_start_pos - pivot_world
        r_pivot_to_base_start[2] = 0  # Project to horizontal plane
        radius = np.linalg.norm(r_pivot_to_base_start[:2])
        
        # Scale down radius for workspace safety
        radius = radius * self.motion_radius_scale
        
        if radius > 0.01:
            # Normalize radius vector
            r_normalized = r_pivot_to_base_start / np.linalg.norm(r_pivot_to_base_start[:2])
            r_normalized[2] = 0
            
            # Current rotation angle in segment
            theta = segment_phase * self.yaw_per_segment
            
            # Rotate radius vector by theta to get new base position
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            rotated_x = r_normalized[0] * cos_theta - r_normalized[1] * sin_theta
            rotated_y = r_normalized[0] * sin_theta + r_normalized[1] * cos_theta
            
            # New base position
            new_base_pos = pivot_world + radius * np.array([rotated_x, rotated_y, 0.0])
            new_base_pos[2] = base_start_pos[2]  # Maintain height
            
            # Compute yaw rotation
            yaw_start = quat_to_yaw(base_start_quat)
            yaw_current = yaw_start + theta
            new_quat = yaw_to_quat(yaw_current)
            
            # Smooth update using exponential smoothing to avoid discontinuities
            alpha = min(1.0, dt * 10.0)  # Smoothing factor
            self.root_pos = (1 - alpha) * self.root_pos + alpha * new_base_pos
            self.root_quat = new_quat  # Direct assignment for orientation
        else:
            # Minimal radius, just rotate in place
            yaw_start = quat_to_yaw(base_start_quat)
            theta = segment_phase * self.yaw_per_segment
            yaw_current = yaw_start + theta
            self.root_quat = yaw_to_quat(yaw_current)

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for each leg.
        - Pivot leg: maintains fixed world position
        - Dragging legs: move in arcs around pivot in world frame, then convert to body frame
        """
        pivot_leg = self.get_current_pivot_leg(phase)
        segment_phase = self.get_segment_phase(phase)
        segment_idx = self.get_segment_index(phase)
        
        segment_key = f"{segment_idx}"
        
        # Ensure segment start positions are initialized
        if segment_key not in self.segment_start_world_positions:
            self.segment_start_world_positions[segment_key] = {}
            for ln in self.leg_names:
                foot_body = self.base_feet_pos_body[ln].copy()
                foot_world = body_to_world_position(foot_body, self.root_pos, self.root_quat)
                self.segment_start_world_positions[segment_key][ln] = foot_world.copy()
        
        # Get pivot leg world position (fixed)
        pivot_world = self.segment_start_world_positions[segment_key][pivot_leg].copy()
        
        if leg_name == pivot_leg:
            # Pivot leg stays fixed in world frame
            foot_world = pivot_world.copy()
        else:
            # Dragging leg: compute arc motion in world frame around pivot
            foot_start_world = self.segment_start_world_positions[segment_key][leg_name].copy()
            
            # Vector from pivot to foot at segment start (xy plane)
            r_pivot_to_foot = foot_start_world - pivot_world
            r_pivot_to_foot[2] = 0
            foot_radius = np.linalg.norm(r_pivot_to_foot[:2])
            
            if foot_radius > 0.01:
                # Normalize
                r_normalized = r_pivot_to_foot / foot_radius
                
                # Rotate by segment angle to create arc motion
                # Use reduced rotation to keep legs within workspace
                theta = segment_phase * self.yaw_per_segment * self.motion_radius_scale
                
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                rotated_x = r_normalized[0] * cos_theta - r_normalized[1] * sin_theta
                rotated_y = r_normalized[0] * sin_theta + r_normalized[1] * cos_theta
                
                # New foot world position
                foot_world = pivot_world + foot_radius * np.array([rotated_x, rotated_y, 0.0])
                foot_world[2] = foot_start_world[2]  # Maintain ground contact height
            else:
                foot_world = foot_start_world.copy()
        
        # Convert world position to body frame
        R = quat_to_rotation_matrix(self.root_quat)
        foot_body = R.T @ (foot_world - self.root_pos)
        
        return foot_body