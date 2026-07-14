from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_CARTWHEEL_SIDE_TRAVEL_MotionGenerator(BaseMotionGenerator):
    """
    Lateral cartwheel motion with 360° roll and sideways travel.
    
    - Base performs continuous roll about longitudinal axis (360° per cycle)
    - Base translates laterally (positive y direction)
    - Right legs (FR, RR) serve as pivot during phase 0.0-0.55
    - Left legs (FL, RL) serve as pivot during phase 0.35-1.0
    - Overlap periods ensure continuous contact throughout cartwheel
    - Leg trajectories are circular arcs in body frame
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.4
        
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        self.roll_rate = 2.0 * np.pi * self.freq
        self.lateral_velocity = 0.3
        
        self.swing_radius = 0.22
        self.swing_height_max = 0.12
        
        # Extended overlap for contact continuity
        self.right_contact_start = 0.0
        self.right_contact_end = 0.55
        self.left_contact_start = 0.35
        self.left_contact_end = 1.0
        
        self.transition_blend = 0.1
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update_base_motion(self, phase, dt):
        """
        Update base with constant roll rate and lateral velocity.
        Base height modulates to maintain contact feasibility.
        """
        roll_rate = self.roll_rate
        vy = self.lateral_velocity
        
        # Base height modulation synchronized to roll phase
        # Lowers toward contact side to keep feet reachable
        base_height_offset = -0.08 * np.sin(2.0 * np.pi * phase)
        vz = -0.08 * 2.0 * np.pi * self.freq * np.cos(2.0 * np.pi * phase)
        
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
        
        Right legs (FR, RR): stance 0.0-0.55, swing 0.55-1.0
        Left legs (FL, RL): swing 0.0-0.35, stance 0.35-1.0
        
        Transitions are blended to ensure smooth contact transfer.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        if is_right:
            stance_start = self.right_contact_start
            stance_end = self.right_contact_end
            swing_start = self.right_contact_end
            swing_end = 1.0
            
            if phase < stance_end:
                foot_stance = self._compute_stance_trajectory(base_pos, phase, stance_start, stance_end, is_left=False)
                if phase > stance_end - self.transition_blend:
                    # Blend to swing at end of stance
                    foot_swing = self._compute_swing_trajectory(base_pos, phase, swing_start, swing_end + (1.0 - swing_end), is_left=False)
                    blend_factor = (phase - (stance_end - self.transition_blend)) / self.transition_blend
                    blend_factor = np.clip(blend_factor, 0.0, 1.0)
                    blend_factor = smooth_step(blend_factor)
                    foot = (1.0 - blend_factor) * foot_stance + blend_factor * foot_swing
                else:
                    foot = foot_stance
            else:
                foot = self._compute_swing_trajectory(base_pos, phase, swing_start, swing_end + (1.0 - swing_end), is_left=False)
        
        elif is_left:
            swing_start = 0.0
            swing_end = self.left_contact_start
            stance_start = self.left_contact_start
            stance_end = self.left_contact_end
            
            if phase < swing_end:
                foot = self._compute_swing_trajectory(base_pos, phase, swing_start, swing_end, is_left=True)
            elif phase < stance_start + self.transition_blend:
                # Blend from swing to stance
                foot_swing = self._compute_swing_trajectory(base_pos, phase, swing_start, swing_end, is_left=True)
                foot_stance = self._compute_stance_trajectory(base_pos, phase, stance_start, stance_end, is_left=True)
                blend_factor = (phase - stance_start) / self.transition_blend
                blend_factor = np.clip(blend_factor, 0.0, 1.0)
                blend_factor = smooth_step(blend_factor)
                foot = (1.0 - blend_factor) * foot_swing + blend_factor * foot_stance
            else:
                foot = self._compute_stance_trajectory(base_pos, phase, stance_start, stance_end, is_left=True)
        else:
            foot = base_pos
        
        return foot
    
    def _compute_stance_trajectory(self, base_pos, phase, start_phase, end_phase, is_left):
        """
        Stance trajectory maintains ground contact while base rolls.
        Foot rotates in body frame y-z plane with constrained radius and angle range.
        """
        progress = (phase - start_phase) / (end_phase - start_phase)
        progress = np.clip(progress, 0.0, 1.0)
        progress = smooth_step(progress)
        
        # Reduced angular range for workspace feasibility
        if is_left:
            angle_start = -np.pi / 6
            angle_end = np.pi / 2
        else:
            angle_start = -np.pi / 2
            angle_end = np.pi / 6
        
        angle = angle_start + progress * (angle_end - angle_start)
        
        # Use reduced radius to stay within workspace
        base_radius = np.sqrt(base_pos[1]**2 + base_pos[2]**2)
        radius = min(base_radius, 0.25)
        
        foot = base_pos.copy()
        foot[1] = radius * np.sin(angle)
        foot[2] = -abs(base_pos[2]) + radius * (1.0 + np.cos(angle)) * 0.5
        
        # Ensure ground clearance minimum
        foot[2] = max(foot[2], -abs(base_pos[2]) * 0.95)
        
        return foot
    
    def _compute_swing_trajectory(self, base_pos, phase, start_phase, end_phase, is_left):
        """
        Swing trajectory with reduced radius and height for workspace feasibility.
        Arc positioned to start and end near ground level.
        """
        progress = (phase - start_phase) / (end_phase - start_phase)
        progress = np.clip(progress, 0.0, 1.0)
        progress = smooth_step(progress)
        
        # Constrained angular range for reachability
        if is_left:
            angle_start = np.pi / 6
            angle_end = 5.0 * np.pi / 6
        else:
            angle_start = 5.0 * np.pi / 6
            angle_end = np.pi / 6
        
        angle = angle_start + progress * (angle_end - angle_start)
        
        foot = base_pos.copy()
        
        # Reduced swing radius
        swing_radius = self.swing_radius
        
        # Circular arc centered above base position
        foot[1] = swing_radius * np.sin(angle)
        
        # Base z position offset to start near ground
        z_offset = -abs(base_pos[2]) + swing_radius * 0.3
        foot[2] = z_offset + swing_radius * np.cos(angle)
        
        # Reduced height boost during mid-swing
        height_boost = self.swing_height_max * np.sin(np.pi * progress)
        foot[2] += height_boost
        
        # Ensure swing starts and ends low for ground contact transition
        if progress < 0.15 or progress > 0.85:
            edge_factor = min(progress / 0.15, (1.0 - progress) / 0.15)
            edge_factor = np.clip(edge_factor, 0.0, 1.0)
            foot[2] = foot[2] * edge_factor + z_offset * (1.0 - edge_factor)
        
        return foot


def smooth_step(t):
    """Smooth interpolation function for blending."""
    return t * t * (3.0 - 2.0 * t)