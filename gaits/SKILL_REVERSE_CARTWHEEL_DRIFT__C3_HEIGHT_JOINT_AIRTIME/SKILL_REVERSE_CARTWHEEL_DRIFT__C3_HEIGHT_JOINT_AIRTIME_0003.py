from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERSE_CARTWHEEL_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Reverse cartwheel drift: continuous backward translation with 360-degree roll per cycle.
    
    Motion characteristics:
    - Base completes full roll rotation (360 degrees) over one phase cycle
    - Sustained backward velocity throughout rotation
    - Lateral leg pairs alternate: left legs (FL, RL) stance [0.0-0.2], right legs (FR, RR) stance [0.8-1.0]
    - Legs trace elliptical arcs in body frame during swing phases with emphasis on downward extension
    - Inverted phase [0.3-0.7] with minimal ground contact, relying on dynamic momentum
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for complex cartwheel motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Cartwheel motion parameters
        self.roll_rate = 2.0 * np.pi * self.freq  # 360 degrees per cycle
        self.backward_velocity = -0.8  # Negative x for backward motion
        
        # Leg trajectory parameters - further reduced and modified for elliptical arc
        self.leg_arc_radius_lateral = 0.13  # Lateral component - reduced from 0.20
        self.leg_arc_radius_vertical = 0.22  # Vertical component - larger for downward emphasis
        self.stance_extension = 0.09  # Reduced from 0.10
        self.vertical_offset_swing = -0.02  # Negative to bias downward
        self.stance_vertical_offset = -0.08  # Reduced flexion from -0.12
        
        # Arc angle limits - maintain reduced sweep
        self.arc_angle_range = 2.0 * np.pi / 3.0  # 120 degrees
        
        # Vertical velocity adjustments - reduced and balanced
        self.vz_ascent = 0.08
        self.vz_descent = -0.08
        
        # Base height maintenance
        self.target_base_height = 0.30  # Target height in world frame
        self.height_correction_gain = 0.3  # Proportional gain for height correction
        
        # Phase timing for contact transitions with wider blend zones
        self.left_stance_end = 0.2
        self.left_swing_end = 0.8
        self.right_swing_start = 0.0
        self.right_swing_end = 0.7
        self.right_stance_start = 0.8
        
        # Blend zone width - increased for smoother transitions
        self.blend_width = 0.12
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.target_base_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base with continuous backward velocity and constant roll rate.
        Includes active height maintenance to prevent sinking during inverted phase.
        """
        # Constant backward velocity
        vx = self.backward_velocity
        
        # Base height correction to prevent sinking
        height_error = self.target_base_height - self.root_pos[2]
        vz_correction = self.height_correction_gain * height_error
        
        # Phase-based vertical velocity adjustments
        if phase < 0.25:
            # Ascent phase: slight upward velocity
            vz_phase = self.vz_ascent
        elif 0.5 <= phase < 0.75:
            # Descent phase: slight downward velocity
            vz_phase = self.vz_descent
        else:
            # Inverted and recovery phases: neutral
            vz_phase = 0.0
        
        # During inverted phase, add extra upward bias to counteract sinking
        if 0.3 <= phase <= 0.7:
            vz_inverted_bias = 0.05
        else:
            vz_inverted_bias = 0.0
        
        # Combine vertical velocity components
        vz = vz_phase + vz_correction + vz_inverted_bias
        
        # Constant roll rate for 360-degree rotation per cycle
        roll_rate = self.roll_rate
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
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
        Compute foot position in body frame based on leg group and phase.
        Uses elliptical arc with downward emphasis to reduce knee extension requirements.
        
        Left legs (FL, RL): stance [0.0-0.2], swing overhead [0.2-0.8], return [0.8-1.0]
        Right legs (FR, RR): swing overhead [0.0-0.7], transition [0.7-0.8], stance [0.8-1.0]
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg side
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        if is_left:
            # Left leg group: FL, RL
            if phase < self.left_stance_end:
                # Stance phase with blending at exit
                if phase < self.left_stance_end - self.blend_width:
                    foot = self._compute_stance_position(base_pos, side='left')
                else:
                    # Blend from stance to arc entry
                    blend_phase = (phase - (self.left_stance_end - self.blend_width)) / self.blend_width
                    stance_foot = self._compute_stance_position(base_pos, side='left')
                    arc_foot = self._compute_elliptical_arc(base_pos, 0.0, side='left')
                    foot = self._smooth_blend(stance_foot, arc_foot, blend_phase)
            elif phase < self.left_swing_end:
                # Swing phase: elliptical arc
                if phase < self.left_stance_end + self.blend_width:
                    # Early blend into arc
                    swing_phase = 0.0
                elif phase > self.left_swing_end - self.blend_width:
                    # Late blend out of arc
                    swing_phase = 1.0
                else:
                    swing_phase = (phase - (self.left_stance_end + self.blend_width)) / \
                                 (self.left_swing_end - self.left_stance_end - 2 * self.blend_width)
                    swing_phase = np.clip(swing_phase, 0.0, 1.0)
                foot = self._compute_elliptical_arc(base_pos, swing_phase, side='left')
            else:
                # Return phase: descend and reposition
                return_phase = (phase - self.left_swing_end) / (1.0 - self.left_swing_end)
                foot = self._compute_return_trajectory(base_pos, return_phase, side='left')
        
        elif is_right:
            # Right leg group: FR, RR - use slightly smaller arc for longer swing duration
            arc_scale = 0.92  # 8% smaller arc for right legs
            
            if phase < self.right_swing_end:
                # Swing phase: elliptical arc
                if phase < self.blend_width:
                    # Blend from initial stance to arc
                    blend_phase = phase / self.blend_width
                    stance_foot = self._compute_stance_position(base_pos, side='right')
                    arc_foot = self._compute_elliptical_arc(base_pos, 0.0, side='right', scale=arc_scale)
                    foot = self._smooth_blend(stance_foot, arc_foot, blend_phase)
                elif phase > self.right_swing_end - self.blend_width:
                    # Late blend
                    swing_phase = 1.0
                    foot = self._compute_elliptical_arc(base_pos, swing_phase, side='right', scale=arc_scale)
                else:
                    swing_phase = (phase - self.blend_width) / (self.right_swing_end - 2 * self.blend_width)
                    swing_phase = np.clip(swing_phase, 0.0, 1.0)
                    foot = self._compute_elliptical_arc(base_pos, swing_phase, side='right', scale=arc_scale)
            elif phase < self.right_stance_start:
                # Transition phase: descend toward ground
                transition_phase = (phase - self.right_swing_end) / (self.right_stance_start - self.right_swing_end)
                foot = self._compute_descent_trajectory(base_pos, transition_phase, side='right', scale=arc_scale)
            else:
                # Stance phase
                foot = self._compute_stance_position(base_pos, side='right')
        else:
            foot = base_pos
        
        return foot

    def _smooth_blend(self, pos1, pos2, alpha):
        """Smooth interpolation between two positions using cubic easing."""
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3  # Smoothstep function
        return pos1 + (pos2 - pos1) * alpha_smooth

    def _compute_stance_position(self, base_pos, side):
        """
        Compute foot position during stance phase.
        Moderate flexion with reduced lateral extension.
        """
        foot = base_pos.copy()
        
        # Lateral extension based on side - reduced for safety
        lateral_sign = -1.0 if side == 'left' else 1.0
        foot[1] += lateral_sign * self.stance_extension
        
        # Moderate downward extension for flexed stance
        foot[2] += self.stance_vertical_offset
        
        return foot

    def _compute_elliptical_arc(self, base_pos, swing_phase, side, scale=1.0):
        """
        Compute foot position during overhead arc swing using elliptical trajectory.
        Emphasizes downward extension to allow knee flexion.
        
        The ellipse has:
        - Smaller lateral radius (horizontal extension)
        - Larger vertical radius (downward drop)
        This creates a "drooping" arc that keeps knees more flexed.
        """
        foot = base_pos.copy()
        
        # Reduced arc angle range
        half_range = self.arc_angle_range / 2.0
        
        if side == 'left':
            # Left legs: arc from left to right
            arc_angle = -half_range + self.arc_angle_range * swing_phase
        else:
            # Right legs: arc from right to left
            arc_angle = half_range - self.arc_angle_range * swing_phase
        
        # Elliptical arc trajectory with downward emphasis
        # Lateral component (smaller radius)
        lateral_radius = self.leg_arc_radius_lateral * scale
        foot[1] = base_pos[1] + lateral_radius * np.cos(arc_angle)
        
        # Vertical component (larger radius, negative for downward)
        # Use negative sine to create downward arc
        vertical_radius = self.leg_arc_radius_vertical * scale
        foot[2] = base_pos[2] - vertical_radius * np.abs(np.sin(arc_angle)) + self.vertical_offset_swing
        
        return foot

    def _compute_return_trajectory(self, base_pos, return_phase, side):
        """
        Compute foot position during return phase (left legs after overhead arc).
        Smooth transition from overhead position back to stance preparation.
        """
        foot = base_pos.copy()
        
        # End of arc position
        half_range = self.arc_angle_range / 2.0
        end_arc_angle = half_range  # Right side for left legs
        
        lateral_radius = self.leg_arc_radius_lateral
        vertical_radius = self.leg_arc_radius_vertical
        
        start_y = base_pos[1] + lateral_radius * np.cos(end_arc_angle)
        start_z = base_pos[2] - vertical_radius * np.abs(np.sin(end_arc_angle)) + self.vertical_offset_swing
        
        # Target stance position
        end_y = base_pos[1] - self.stance_extension
        end_z = base_pos[2] + self.stance_vertical_offset
        
        # Smooth interpolation with cubic easing
        alpha = 3 * return_phase**2 - 2 * return_phase**3
        foot[1] = start_y + (end_y - start_y) * alpha
        foot[2] = start_z + (end_z - start_z) * alpha
        
        return foot

    def _compute_descent_trajectory(self, base_pos, transition_phase, side, scale=1.0):
        """
        Compute foot position during descent transition (right legs before stance).
        Smooth transition from overhead position to ground contact.
        """
        foot = base_pos.copy()
        
        # End of arc position
        half_range = self.arc_angle_range / 2.0
        end_arc_angle = -half_range  # Left side for right legs
        
        lateral_radius = self.leg_arc_radius_lateral * scale
        vertical_radius = self.leg_arc_radius_vertical * scale
        
        start_y = base_pos[1] + lateral_radius * np.cos(end_arc_angle)
        start_z = base_pos[2] - vertical_radius * np.abs(np.sin(end_arc_angle)) + self.vertical_offset_swing
        
        # Target stance position
        end_y = base_pos[1] + self.stance_extension
        end_z = base_pos[2] + self.stance_vertical_offset
        
        # Smooth interpolation with parabolic descent for natural landing
        alpha = 3 * transition_phase**2 - 2 * transition_phase**3
        foot[1] = start_y + (end_y - start_y) * alpha
        foot[2] = start_z + (end_z - start_z) * (alpha ** 0.85)
        
        return foot