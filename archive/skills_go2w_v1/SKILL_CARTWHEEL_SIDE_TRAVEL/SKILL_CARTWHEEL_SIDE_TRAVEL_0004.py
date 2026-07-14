from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_CARTWHEEL_SIDE_TRAVEL_MotionGenerator(BaseMotionGenerator):
    """
    Lateral cartwheel motion with 360° roll and sideways travel.
    
    - Base performs continuous roll about longitudinal axis (360° per cycle)
    - Base translates laterally (positive y direction)
    - Right legs (FR, RR) serve as pivot during phase 0.0-0.45
    - Left legs (FL, RL) serve as pivot during phase 0.4-0.95
    - Leg trajectories are circular arcs synchronized with roll
    - Base height modulates to maintain ground contact feasibility
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.4
        
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        self.roll_rate = 2.0 * np.pi * self.freq
        self.lateral_velocity = 0.25
        
        # Reduced for joint limit compliance
        self.swing_radius = 0.17
        self.swing_height_world = 0.11
        
        # Extended contact phases with overlap to prevent simultaneous airtime
        self.right_contact_start = 0.0
        self.right_contact_end = 0.45
        self.left_contact_start = 0.4
        self.left_contact_end = 0.95
        
        self.transition_blend = 0.05
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update_base_motion(self, phase, dt):
        """
        Update base with constant roll rate and lateral velocity.
        Base height modulates to maintain contact feasibility during roll.
        """
        roll_rate = self.roll_rate
        vy = self.lateral_velocity
        
        # Roll angle for current phase
        roll_angle = 2.0 * np.pi * phase
        
        # Reduced base height modulation amplitude to maintain envelope compliance
        base_height_amplitude = 0.13
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
        
        Right legs (FR, RR): stance 0.0-0.45, swing 0.45-1.0
        Left legs (FL, RL): swing 0.0-0.4, stance 0.4-0.95, swing 0.95-1.0
        
        Overlap region 0.4-0.45 ensures continuous ground contact.
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
                # Second swing phase (handles wrap to next cycle)
                foot = self._compute_swing_trajectory(base_pos, phase, 
                                                     self.left_contact_end, 
                                                     1.0, roll_angle, is_left=True)
        else:
            foot = base_pos
        
        # Apply roll-dependent workspace constraint
        foot = self._constrain_to_workspace(foot, roll_angle)
        
        return foot
    
    def _compute_ground_locked_stance(self, base_pos, phase, start_phase, end_phase, roll_angle, is_left):
        """
        Stance trajectory maintains ground contact in world frame.
        Computes body-frame position that results in world z ≈ ground level.
        """
        progress = (phase - start_phase) / (end_phase - start_phase)
        progress = np.clip(progress, 0.0, 1.0)
        progress = smooth_step(progress)
        
        # Lateral offset for travel and stability (reduced for workspace compliance)
        lateral_offset = 0.12 if is_left else -0.12
        
        # Ground contact point moves during stance for smooth travel
        forward_shift = 0.06 * (progress - 0.5)
        
        # World frame target (ground level)
        foot_world = np.array([
            forward_shift,
            lateral_offset,
            0.0  # Ground level
        ])
        
        # Transform to body frame accounting for roll
        cos_r = np.cos(-roll_angle)
        sin_r = np.sin(-roll_angle)
        
        foot_body = base_pos.copy()
        foot_body[0] = foot_world[0]
        foot_body[1] = cos_r * foot_world[1] - sin_r * foot_world[2]
        foot_body[2] = sin_r * foot_world[1] + cos_r * foot_world[2]
        
        # Conservative workspace constraint
        foot_body = self._constrain_to_workspace(foot_body, roll_angle)
        
        return foot_body
    
    def _compute_swing_trajectory(self, base_pos, phase, start_phase, end_phase, roll_angle, is_left):
        """
        Swing trajectory with circular arc ensuring clearance while respecting joint limits.
        Reduced arc geometry for kinematic feasibility.
        """
        progress = (phase - start_phase) / (end_phase - start_phase)
        progress = np.clip(progress, 0.0, 1.0)
        progress = smooth_step(progress)
        
        # Circular arc in body frame y-z plane (reduced angular range)
        if is_left:
            angle_start = -np.pi * 0.3
            angle_end = np.pi * 0.6
        else:
            angle_start = np.pi * 0.6
            angle_end = np.pi * 1.3
        
        angle = angle_start + progress * (angle_end - angle_start)
        
        foot = base_pos.copy()
        
        swing_radius = self.swing_radius
        
        # Circular arc centered with lateral offset
        lateral_center = 0.08 if is_left else -0.08
        foot[1] = lateral_center + swing_radius * np.sin(angle)
        foot[2] = swing_radius * (np.cos(angle) - 1.0)
        
        # Additional height boost during mid-swing (reduced magnitude)
        height_boost_mag = self.swing_height_world * np.sin(np.pi * progress)
        
        # Roll-dependent height boost scaling to prevent extreme joint angles
        roll_factor = np.cos(roll_angle)
        height_scale = 0.4 + 0.6 * np.abs(roll_factor)  # Reduce boost when inverted
        height_boost_mag *= height_scale
        
        # Transform height boost to account for roll orientation
        cos_r = np.cos(roll_angle)
        sin_r = np.sin(roll_angle)
        
        # Add height in direction that increases world z
        foot[1] += height_boost_mag * sin_r
        foot[2] += height_boost_mag * cos_r
        
        # Smooth edge blending for continuous transitions
        edge_blend = 0.15
        if progress < edge_blend:
            edge_factor = progress / edge_blend
            edge_factor = smooth_step(edge_factor)
            # Blend from stance-like position
            foot_start = base_pos.copy()
            foot_start[1] = lateral_center
            foot_start[2] = -0.08
            foot = edge_factor * foot + (1.0 - edge_factor) * foot_start
        elif progress > 1.0 - edge_blend:
            edge_factor = (1.0 - progress) / edge_blend
            edge_factor = smooth_step(edge_factor)
            # Blend to stance-like position
            foot_end = base_pos.copy()
            foot_end[1] = lateral_center
            foot_end[2] = -0.08
            foot = edge_factor * foot + (1.0 - edge_factor) * foot_end
        
        # Apply workspace constraint
        foot = self._constrain_to_workspace(foot, roll_angle)
        
        return foot
    
    def _constrain_to_workspace(self, foot_body, roll_angle):
        """
        Roll-dependent workspace constraint to maintain joint limit compliance.
        More conservative limits applied during high roll angles.
        """
        # Base radial constraint
        max_radius_base = 0.24
        
        # Roll-dependent radius reduction (more conservative when inverted)
        roll_severity = np.abs(np.sin(roll_angle))  # Max at 90° and 270°
        max_radius = max_radius_base * (1.0 - 0.2 * roll_severity)
        
        # Y-Z plane constraint
        yz_dist = np.sqrt(foot_body[1]**2 + foot_body[2]**2)
        if yz_dist > max_radius:
            scale = max_radius / yz_dist
            foot_body[1] *= scale
            foot_body[2] *= scale
        
        # Z-axis constraint (prevent extreme downward extension)
        max_z_down = -0.25
        max_z_up = 0.15
        foot_body[2] = np.clip(foot_body[2], max_z_down, max_z_up)
        
        # X-axis constraint (forward/backward reach)
        max_x = 0.2
        foot_body[0] = np.clip(foot_body[0], -max_x, max_x)
        
        return foot_body


def smooth_step(t):
    """Smooth interpolation function for blending."""
    return t * t * (3.0 - 2.0 * t)