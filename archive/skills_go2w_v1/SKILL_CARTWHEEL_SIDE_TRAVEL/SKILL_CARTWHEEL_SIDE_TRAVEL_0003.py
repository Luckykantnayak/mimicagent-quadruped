from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_CARTWHEEL_SIDE_TRAVEL_MotionGenerator(BaseMotionGenerator):
    """
    Lateral cartwheel motion with 360° roll and sideways travel.
    
    - Base performs continuous roll about longitudinal axis (360° per cycle)
    - Base translates laterally (positive y direction)
    - Right legs (FR, RR) serve as pivot during phase 0.0-0.3
    - Left legs (FL, RL) serve as pivot during phase 0.5-0.8
    - Leg trajectories are circular arcs synchronized with roll
    - Base height modulates to maintain ground contact feasibility
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.4
        
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        self.roll_rate = 2.0 * np.pi * self.freq
        self.lateral_velocity = 0.25
        
        self.swing_radius = 0.24
        self.swing_height_world = 0.18
        
        # Contact phases aligned with roll geometry
        self.right_contact_start = 0.0
        self.right_contact_end = 0.3
        self.left_contact_start = 0.5
        self.left_contact_end = 0.8
        
        self.transition_blend = 0.06
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update_base_motion(self, phase, dt):
        """
        Update base with constant roll rate and lateral velocity.
        Base height modulates significantly to maintain contact feasibility during roll.
        """
        roll_rate = self.roll_rate
        vy = self.lateral_velocity
        
        # Roll angle for current phase
        roll_angle = 2.0 * np.pi * phase
        
        # Base height modulation amplitude increased for roll geometry
        # Lowers significantly when rolled to allow feet to reach ground
        base_height_amplitude = 0.22
        base_height_offset = -base_height_amplitude * np.cos(2.0 * np.pi * phase)
        vz = base_height_amplitude * 2.0 * np.pi * self.freq * np.sin(2.0 * np.pi * phase)
        
        self.vel_world = np.array([0.0, vy, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on contact/swing state.
        
        Right legs (FR, RR): stance 0.0-0.3, swing 0.3-1.0
        Left legs (FL, RL): swing 0.0-0.5, stance 0.5-0.8, swing 0.8-1.0
        
        Stance phases locked to ground in world frame, transformed to body frame.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        roll_angle = 2.0 * np.pi * phase
        
        if is_right:
            if phase < self.right_contact_end:
                # Stance phase
                foot = self._compute_ground_locked_stance(base_pos, phase, 
                                                          self.right_contact_start, 
                                                          self.right_contact_end, 
                                                          roll_angle, is_left=False)
                # Blend to swing near end
                if phase > self.right_contact_end - self.transition_blend:
                    foot_swing = self._compute_swing_trajectory(base_pos, phase, 
                                                                self.right_contact_end, 
                                                                1.0, roll_angle, is_left=False)
                    blend_factor = (phase - (self.right_contact_end - self.transition_blend)) / self.transition_blend
                    blend_factor = smooth_step(np.clip(blend_factor, 0.0, 1.0))
                    foot = (1.0 - blend_factor) * foot + blend_factor * foot_swing
            else:
                # Swing phase
                foot = self._compute_swing_trajectory(base_pos, phase, 
                                                     self.right_contact_end, 
                                                     1.0, roll_angle, is_left=False)
        
        elif is_left:
            if phase < self.left_contact_start:
                # First swing phase
                foot = self._compute_swing_trajectory(base_pos, phase, 
                                                     0.0, 
                                                     self.left_contact_start, 
                                                     roll_angle, is_left=True)
            elif phase < self.left_contact_end:
                # Stance phase
                foot = self._compute_ground_locked_stance(base_pos, phase, 
                                                          self.left_contact_start, 
                                                          self.left_contact_end, 
                                                          roll_angle, is_left=True)
                # Blend from swing at start
                if phase < self.left_contact_start + self.transition_blend:
                    foot_swing = self._compute_swing_trajectory(base_pos, phase, 
                                                                0.0, 
                                                                self.left_contact_start, 
                                                                roll_angle, is_left=True)
                    blend_factor = (phase - self.left_contact_start) / self.transition_blend
                    blend_factor = smooth_step(np.clip(blend_factor, 0.0, 1.0))
                    foot = (1.0 - blend_factor) * foot_swing + blend_factor * foot
                # Blend to swing at end
                elif phase > self.left_contact_end - self.transition_blend:
                    foot_swing = self._compute_swing_trajectory(base_pos, phase, 
                                                                self.left_contact_end, 
                                                                1.0, roll_angle, is_left=True)
                    blend_factor = (phase - (self.left_contact_end - self.transition_blend)) / self.transition_blend
                    blend_factor = smooth_step(np.clip(blend_factor, 0.0, 1.0))
                    foot = (1.0 - blend_factor) * foot + blend_factor * foot_swing
            else:
                # Second swing phase
                foot = self._compute_swing_trajectory(base_pos, phase, 
                                                     self.left_contact_end, 
                                                     1.0, roll_angle, is_left=True)
        else:
            foot = base_pos
        
        return foot
    
    def _compute_ground_locked_stance(self, base_pos, phase, start_phase, end_phase, roll_angle, is_left):
        """
        Stance trajectory maintains ground contact in world frame.
        Computes body-frame position that results in world z ≈ ground level.
        """
        progress = (phase - start_phase) / (end_phase - start_phase)
        progress = np.clip(progress, 0.0, 1.0)
        progress = smooth_step(progress)
        
        # Compute target world-frame foot position on ground
        # Lateral offset for travel and stability
        lateral_offset = 0.15 if is_left else -0.15
        
        # Ground contact point moves slightly during stance for smooth travel
        forward_shift = 0.08 * (progress - 0.5)
        
        # World frame target (ground level)
        foot_world = np.array([
            forward_shift,
            lateral_offset,
            0.0  # Ground level
        ])
        
        # Transform to body frame accounting for roll
        # Simplified inverse rotation about x-axis (roll)
        cos_r = np.cos(-roll_angle)
        sin_r = np.sin(-roll_angle)
        
        foot_body = base_pos.copy()
        foot_body[0] = foot_world[0]
        foot_body[1] = cos_r * foot_world[1] - sin_r * foot_world[2]
        foot_body[2] = sin_r * foot_world[1] + cos_r * foot_world[2]
        
        # Constrain to workspace
        max_radius = 0.28
        yz_dist = np.sqrt(foot_body[1]**2 + foot_body[2]**2)
        if yz_dist > max_radius:
            scale = max_radius / yz_dist
            foot_body[1] *= scale
            foot_body[2] *= scale
        
        return foot_body
    
    def _compute_swing_trajectory(self, base_pos, phase, start_phase, end_phase, roll_angle, is_left):
        """
        Swing trajectory with circular arc ensuring world-frame clearance.
        Arc provides sufficient height above ground during inverted phases.
        """
        progress = (phase - start_phase) / (end_phase - start_phase)
        progress = np.clip(progress, 0.0, 1.0)
        progress = smooth_step(progress)
        
        # Circular arc in body frame y-z plane
        if is_left:
            angle_start = -np.pi * 0.35
            angle_end = np.pi * 0.65
        else:
            angle_start = np.pi * 0.65
            angle_end = np.pi * 1.35
        
        angle = angle_start + progress * (angle_end - angle_start)
        
        foot = base_pos.copy()
        
        swing_radius = self.swing_radius
        
        # Circular arc centered with lateral offset
        lateral_center = 0.1 if is_left else -0.1
        foot[1] = lateral_center + swing_radius * np.sin(angle)
        foot[2] = swing_radius * (np.cos(angle) - 1.0)
        
        # Additional height boost during mid-swing for world-frame clearance
        height_boost = self.swing_height_world * np.sin(np.pi * progress)
        
        # Transform height boost to account for roll orientation
        # When inverted, body z points up in world frame
        cos_r = np.cos(roll_angle)
        sin_r = np.sin(roll_angle)
        
        # Add height in direction that increases world z
        foot[1] += height_boost * sin_r
        foot[2] += height_boost * cos_r
        
        # Ensure swing starts and ends low for smooth transitions
        edge_blend = 0.2
        if progress < edge_blend:
            edge_factor = progress / edge_blend
            edge_factor = smooth_step(edge_factor)
            # Blend from ground-like position
            foot_start = base_pos.copy()
            foot_start[1] = lateral_center
            foot_start[2] = -abs(base_pos[2]) * 0.5
            foot = edge_factor * foot + (1.0 - edge_factor) * foot_start
        elif progress > 1.0 - edge_blend:
            edge_factor = (1.0 - progress) / edge_blend
            edge_factor = smooth_step(edge_factor)
            # Blend to ground-like position
            foot_end = base_pos.copy()
            foot_end[1] = lateral_center
            foot_end[2] = -abs(base_pos[2]) * 0.5
            foot = edge_factor * foot + (1.0 - edge_factor) * foot_end
        
        return foot


def smooth_step(t):
    """Smooth interpolation function for blending."""
    return t * t * (3.0 - 2.0 * t)