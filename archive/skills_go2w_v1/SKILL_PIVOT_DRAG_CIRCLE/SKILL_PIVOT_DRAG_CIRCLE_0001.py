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
        self.segment_duration = 1.0 / (3.0 * self.freq)  # Duration of each phase segment
        
        # Circular motion radius around pivot (in body frame)
        self.circle_radius = 0.4
        
        # Angular velocity for constant yaw rate per segment
        self.yaw_rate = self.yaw_per_segment / self.segment_duration

        # Store world positions of pivot legs to maintain fixed position
        self.pivot_world_positions = {}

    def reset(self, root_pos, root_quat):
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0
        self.pivot_world_positions = {}

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
        
        # Constant yaw rate for rotation
        yaw_rate = self.yaw_rate
        
        # Get pivot leg position in body frame
        pivot_pos_body = self.base_feet_pos_body[pivot_leg].copy()
        
        # Compute circular motion velocity in body frame
        # The base should move in a circle around the pivot point
        # Tangential velocity for circular motion: v = ω × r
        # where r is vector from pivot to base center in body frame
        
        # Vector from base center to pivot in body frame
        r_to_pivot = pivot_pos_body.copy()
        r_to_pivot[2] = 0  # Project to horizontal plane
        
        # Radius of circular motion
        radius = np.linalg.norm(r_to_pivot[:2])
        
        if radius > 0.01:
            # Tangent direction (perpendicular to radius, in body frame)
            # For clockwise rotation when viewed from above (positive yaw)
            tangent_x = -r_to_pivot[1]
            tangent_y = r_to_pivot[0]
            tangent_norm = np.sqrt(tangent_x**2 + tangent_y**2)
            
            if tangent_norm > 0.01:
                tangent_x /= tangent_norm
                tangent_y /= tangent_norm
                
                # Tangential velocity magnitude: v = ω * r
                v_tangent = yaw_rate * radius
                
                vx_body = v_tangent * tangent_x
                vy_body = v_tangent * tangent_y
            else:
                vx_body = 0.0
                vy_body = 0.0
        else:
            vx_body = 0.0
            vy_body = 0.0
        
        # Convert body frame velocity to world frame
        R = quat_to_rotation_matrix(self.root_quat)
        vel_body = np.array([vx_body, vy_body, 0.0])
        vel_world = R @ vel_body
        
        # Angular velocity in world frame (pure yaw rotation)
        omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            vel_world,
            omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for each leg.
        - Pivot leg: maintains fixed world position, body-frame position changes as base rotates
        - Dragging legs: move in arcs, body-frame positions evolve smoothly
        """
        pivot_leg = self.get_current_pivot_leg(phase)
        segment_phase = self.get_segment_phase(phase)
        segment_idx = self.get_segment_index(phase)
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        if leg_name == pivot_leg:
            # Pivot leg: fixed in world frame, compute body frame position
            # Store world position at segment start
            segment_key = f"{segment_idx}_{leg_name}"
            if segment_key not in self.pivot_world_positions:
                # Convert current body frame position to world frame at segment start
                self.pivot_world_positions[segment_key] = body_to_world_position(
                    base_pos, self.root_pos, self.root_quat
                )
            
            # Convert fixed world position back to current body frame
            pivot_world = self.pivot_world_positions[segment_key]
            R = quat_to_rotation_matrix(self.root_quat)
            pivot_body = R.T @ (pivot_world - self.root_pos)
            
            return pivot_body
        
        else:
            # Dragging leg: moves in arc around pivot
            # Compute arc trajectory in body frame
            
            # Total rotation accumulated so far in current segment
            segment_yaw = segment_phase * self.yaw_per_segment
            
            # Angle offset for arc motion
            # Each leg traces an arc as the base rotates
            angle_offset = segment_yaw
            
            # Base position rotated by accumulated segment rotation
            cos_a = np.cos(angle_offset)
            sin_a = np.sin(angle_offset)
            
            # Apply rotation around body center to create dragging arc
            # This simulates the foot position changing in body frame as base rotates
            foot_x = base_pos[0] * cos_a - base_pos[1] * sin_a
            foot_y = base_pos[0] * sin_a + base_pos[1] * cos_a
            foot_z = base_pos[2]  # Maintain ground contact (z remains constant)
            
            # Add small adjustments for continuous smooth motion across segments
            # Blend toward next configuration at segment boundaries
            transition_blend = 0.1
            if segment_phase > (1.0 - transition_blend):
                blend_factor = (segment_phase - (1.0 - transition_blend)) / transition_blend
                next_segment_idx = (segment_idx + 1) % 3
                
                # Compute what the position would be at the start of next segment
                if next_segment_idx == 0:
                    next_angle = 0.0
                elif next_segment_idx == 1:
                    next_angle = self.yaw_per_segment
                else:
                    next_angle = 2 * self.yaw_per_segment
                
                cos_next = np.cos(next_angle)
                sin_next = np.sin(next_angle)
                
                foot_x_next = base_pos[0] * cos_next - base_pos[1] * sin_next
                foot_y_next = base_pos[0] * sin_next + base_pos[1] * cos_next
                
                foot_x = (1 - blend_factor) * foot_x + blend_factor * foot_x_next
                foot_y = (1 - blend_factor) * foot_y + blend_factor * foot_y_next
            
            return np.array([foot_x, foot_y, foot_z])