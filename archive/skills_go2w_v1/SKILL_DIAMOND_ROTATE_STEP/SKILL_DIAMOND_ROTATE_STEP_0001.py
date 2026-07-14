from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_DIAMOND_ROTATE_STEP_MotionGenerator(BaseMotionGenerator):
    """
    Diamond-rotate-step gait: all four legs trace synchronized diamond patterns
    on the ground while the base translates forward and rotates 180 degrees per cycle.
    
    - All legs remain in continuous ground contact (stance-only gait)
    - Each leg traces a diamond with 4 vertices: front, side, rear, opposite side
    - Base applies constant forward velocity and constant yaw rate
    - One full cycle (phase 0->1) completes one diamond loop and 180-degree rotation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Hz, slower frequency for stance-only gait
        
        # Base foot positions (nominal center of diamond)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Diamond geometry parameters
        self.diamond_length_forward = 0.12   # forward extent from base
        self.diamond_length_rear = 0.10      # rearward extent from base
        self.diamond_width_lateral = 0.13    # lateral extent from base
        
        # Base velocities
        self.forward_velocity = 0.15  # m/s, modest forward speed
        self.yaw_rate = np.pi         # rad/s, integrates to 180 deg per cycle (period = 2.0s at freq=0.5)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Precompute diamond vertices for each leg in body frame
        self.diamond_vertices = self._compute_diamond_vertices()

    def _compute_diamond_vertices(self):
        """
        Compute the four vertices of the diamond for each leg in body frame.
        Vertices are ordered: front (0), side (1), rear (2), opposite_side (3)
        
        For FL and RL: side = left, opposite_side = right
        For FR and RR: side = right, opposite_side = left
        """
        vertices = {}
        
        for leg_name in self.leg_names:
            base_pos = self.base_feet_pos_body[leg_name].copy()
            
            # Determine lateral direction based on leg name
            if leg_name.startswith('FL') or leg_name.startswith('RL'):
                # Left legs: side vertex to the left (+y), opposite to the right (-y)
                lateral_sign = 1.0
            else:
                # Right legs: side vertex to the right (-y), opposite to the left (+y)
                lateral_sign = -1.0
            
            # Define four vertices relative to base position
            # Vertex 0: front (extended forward, neutral lateral)
            v0 = base_pos + np.array([self.diamond_length_forward, 0.0, 0.0])
            
            # Vertex 1: side (neutral fore-aft, extended lateral)
            v1 = base_pos + np.array([0.0, lateral_sign * self.diamond_width_lateral, 0.0])
            
            # Vertex 2: rear (extended rearward, neutral lateral)
            v2 = base_pos + np.array([-self.diamond_length_rear, 0.0, 0.0])
            
            # Vertex 3: opposite side (neutral fore-aft, extended opposite lateral)
            v3 = base_pos + np.array([0.0, -lateral_sign * self.diamond_width_lateral, 0.0])
            
            vertices[leg_name] = [v0, v1, v2, v3]
        
        return vertices

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity and constant yaw rate.
        Over one full cycle, base moves forward and rotates 180 degrees clockwise.
        """
        # Constant forward velocity in world frame x-direction
        vx = self.forward_velocity
        
        # Constant positive yaw rate (clockwise when viewed from above)
        yaw_rate = self.yaw_rate
        
        self.vel_world = np.array([vx, 0.0, 0.0])
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
        Compute foot position by interpolating through diamond vertices.
        
        Phase mapping:
        [0.00, 0.25]: vertex 0 -> vertex 1 (front to side)
        [0.25, 0.50]: vertex 1 -> vertex 2 (side to rear)
        [0.50, 0.75]: vertex 2 -> vertex 3 (rear to opposite side)
        [0.75, 1.00]: vertex 3 -> vertex 0 (opposite side to front)
        """
        vertices = self.diamond_vertices[leg_name]
        
        # Determine current segment and local progress
        if phase < 0.25:
            # Segment 0->1
            v_start = vertices[0]
            v_end = vertices[1]
            local_phase = phase / 0.25
        elif phase < 0.5:
            # Segment 1->2
            v_start = vertices[1]
            v_end = vertices[2]
            local_phase = (phase - 0.25) / 0.25
        elif phase < 0.75:
            # Segment 2->3
            v_start = vertices[2]
            v_end = vertices[3]
            local_phase = (phase - 0.5) / 0.25
        else:
            # Segment 3->0
            v_start = vertices[3]
            v_end = vertices[0]
            local_phase = (phase - 0.75) / 0.25
        
        # Smooth interpolation using sinusoidal easing for smoother transitions
        smooth_phase = 0.5 * (1.0 - np.cos(np.pi * local_phase))
        
        # Linear interpolation between vertices
        foot_pos = v_start + smooth_phase * (v_end - v_start)
        
        return foot_pos