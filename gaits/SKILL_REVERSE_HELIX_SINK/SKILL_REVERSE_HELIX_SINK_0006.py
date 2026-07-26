from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_REVERSE_HELIX_SINK_MotionGenerator(BaseMotionGenerator):
    """
    Reverse Helical Descent Motion Generator.
    
    The robot executes a continuous backward translation while rotating 
    counter-clockwise and descending in height, tracing a helical spiral 
    path downward over one full phase cycle. All four feet maintain 
    ground contact throughout.
    
    Key approach:
    - Feet maintain world-frame ground contact (z=0)
    - Body-frame foot positions computed by inverse-transforming world positions
    - Base descends via negative vz while rotating and moving backward
    - "Leg compression" emerges naturally as base lowers toward grounded feet
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0  # One full helical cycle per phase period
        
        # Base foot positions (BODY frame at initialization)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.backward_velocity = -0.3  # Constant backward velocity (m/s)
        self.yaw_rate_total = 2 * np.pi  # 360 degrees per cycle
        self.descent_total = 0.10  # Total vertical descent over cycle (m)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Initialize world-frame foot positions (grounded at z=0)
        self.world_feet_pos = {}
        for leg_name in self.leg_names:
            body_pos = self.base_feet_pos_body[leg_name]
            # Transform initial body-frame position to world-frame
            world_pos = self.transform_body_to_world(body_pos, self.root_pos, self.root_quat)
            # Force z to ground level
            world_pos[2] = 0.0
            self.world_feet_pos[leg_name] = world_pos

    def transform_body_to_world(self, pos_body, root_pos, root_quat):
        """Transform position from body frame to world frame."""
        # Rotate body-frame position by root orientation
        pos_rotated = quat_rotate(root_quat, pos_body)
        # Translate by root position
        pos_world = pos_rotated + root_pos
        return pos_world

    def transform_world_to_body(self, pos_world, root_pos, root_quat):
        """Transform position from world frame to body frame."""
        # Translate by negative root position
        pos_translated = pos_world - root_pos
        # Rotate by inverse (conjugate) of root quaternion
        root_quat_inv = quat_conjugate(root_quat)
        pos_body = quat_rotate(root_quat_inv, pos_translated)
        return pos_body

    def update_base_motion(self, phase, dt):
        """
        Update base using constant backward velocity, constant yaw rate,
        and phase-varying descent velocity to create helical trajectory.
        """
        # Constant backward velocity in world frame (initial heading direction)
        # Since base rotates, we apply velocity in current body x direction
        # For simplicity, apply constant velocity in initial world-frame backward direction
        # This creates a backward spiral as base rotates
        
        # Constant yaw rate to complete 360° over one cycle
        yaw_rate = self.yaw_rate_total * self.freq
        
        # Time-varying descent velocity
        # Smooth descent profile: ramp up, peak, taper down
        if phase < 0.25:
            # Ease-in (phase 0.0 -> 0.25)
            t_local = phase / 0.25
            vz_scale = 0.5 * (1 - np.cos(np.pi * t_local))
        elif phase < 0.5:
            # Peak descent (phase 0.25 -> 0.5)
            vz_scale = 1.0
        elif phase < 0.75:
            # Ease-out (phase 0.5 -> 0.75)
            t_local = (phase - 0.5) / 0.25
            vz_scale = 0.5 * (1 + np.cos(np.pi * t_local))
        else:
            # Minimal descent at minimum height (phase 0.75 -> 1.0)
            vz_scale = 0.1
        
        # Scale descent velocity to achieve total descent over cycle
        vz_avg = -self.descent_total * self.freq
        vz_peak = 2.0 * vz_avg  # Peak is roughly 2x average for smooth profile
        vz = vz_peak * vz_scale
        
        # Backward velocity in body frame x direction
        # Transform to world frame using current orientation
        vel_body = np.array([self.backward_velocity, 0.0, 0.0])
        vel_world_lateral = quat_rotate(self.root_quat, vel_body)
        
        # Combine lateral (backward) velocity with vertical descent
        self.vel_world = vel_world_lateral + np.array([0.0, 0.0, vz])
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
        Compute foot position in body frame.
        
        Feet maintain world-frame ground contact. Body-frame positions are
        computed by inverse-transforming the world-frame ground positions
        through the current base pose.
        
        As base descends, rotates, and moves backward, feet automatically
        appear to shift in body frame to maintain world-frame ground contact.
        """
        # Get the world-frame foot position (grounded)
        world_foot_pos = self.world_feet_pos[leg_name].copy()
        
        # Allow feet to slide slightly backward to track base backward motion
        # This prevents legs from over-extending as base moves away
        # Slide at a fraction of base backward velocity to maintain feasible reach
        slide_factor = 0.6  # Feet slide at 60% of base backward velocity
        
        # Compute backward displacement in world frame (in initial heading direction)
        # Approximate: use initial forward direction (x-axis in world at phase 0)
        # Since base rotates, we use current body x direction
        backward_direction = quat_rotate(self.root_quat, np.array([1.0, 0.0, 0.0]))
        backward_slide = slide_factor * self.backward_velocity * phase / self.freq
        world_foot_pos[:2] += backward_direction[:2] * backward_slide
        
        # Ensure foot remains at ground level
        world_foot_pos[2] = 0.0
        
        # Transform world-frame foot position to current body frame
        body_foot_pos = self.transform_world_to_body(
            world_foot_pos, 
            self.root_pos, 
            self.root_quat
        )
        
        # Add small lateral spread for stability at low height
        # This is minimal and applied in body frame to widen stance slightly
        if phase > 0.3:
            spread_progress = min(1.0, (phase - 0.3) / 0.4)
            spread_amount = 0.02 * spread_progress  # Max 2cm spread
            
            if 'FL' in leg_name:
                body_foot_pos[1] += spread_amount  # Left
            elif 'FR' in leg_name:
                body_foot_pos[1] -= spread_amount  # Right
            elif 'RL' in leg_name:
                body_foot_pos[1] += spread_amount  # Left
            elif 'RR' in leg_name:
                body_foot_pos[1] -= spread_amount  # Right
        
        return body_foot_pos