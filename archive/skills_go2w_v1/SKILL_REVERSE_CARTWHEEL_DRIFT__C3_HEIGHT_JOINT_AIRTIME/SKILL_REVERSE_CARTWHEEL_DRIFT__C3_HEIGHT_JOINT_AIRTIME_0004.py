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
    - Legs trace small circular arcs in body frame during swing phases
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
        
        # Leg trajectory parameters - conservative values to respect joint limits
        self.leg_arc_radius = 0.11  # Small radius to minimize reach requirement
        self.stance_extension = 0.08  # Minimal lateral extension
        self.vertical_offset_swing = 0.02  # Small positive offset for clearance
        self.stance_vertical_offset = -0.10  # Moderate flexion during stance
        
        # Arc angle limits - reduced to 90 degrees total sweep
        self.arc_angle_range = np.pi / 2.0  # 90 degrees (-45 to +45)
        
        # Vertical velocity adjustments - symmetric and balanced
        self.vz_ascent = 0.08
        self.vz_descent = -0.08
        
        # Phase timing for contact transitions
        self.left_stance_end = 0.2
        self.left_swing_end = 0.8
        self.right_swing_start = 0.0
        self.right_swing_end = 0.7
        self.right_stance_start = 0.8
        
        # Blend zone width - moderate for smooth transitions
        self.blend_width = 0.09
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base with continuous backward velocity and constant roll rate.
        Natural height variation without active correction.
        """
        # Constant backward velocity
        vx = self.backward_velocity
        
        # Phase-based vertical velocity adjustments - symmetric application
        if phase < 0.25:
            # Ascent phase: upward velocity
            vz = self.vz_ascent
        elif 0.5 <= phase < 0.75:
            # Descent phase: downward velocity
            vz = self.vz_descent
        else:
            # Inverted and recovery phases: neutral
            vz = 0.0
        
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
        Uses small circular arcs to minimize joint extension requirements.
        
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
                    arc_foot = self._compute_circular_arc(base_pos, 0.0, side='left')
                    foot = self._smooth_blend(stance_foot, arc_foot, blend_phase)
            elif phase < self.left_swing_end:
                # Swing phase: circular arc
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
                foot = self._compute_circular_arc(base_pos, swing_phase, side='left')
            else:
                # Return phase: descend and reposition
                return_phase = (phase - self.left_swing_end) / (1.0 - self.left_swing_end)
                foot = self._compute_return_trajectory(base_pos, return_phase, side='left')
        
        elif is_right:
            # Right leg group: FR, RR
            if phase < self.right_swing_end:
                # Swing phase: circular arc
                if phase < self.blend_width:
                    # Blend from initial stance to arc
                    blend_phase = phase / self.blend_width
                    stance_foot = self._compute_stance_position(base_pos, side='right')
                    arc_foot = self._compute_circular_arc(base_pos, 0.0, side='right')
                    foot = self._smooth_blend(stance_foot, arc_foot, blend_phase)
                elif phase > self.right_swing_end - self.blend_width:
                    # Late blend
                    swing_phase = 1.0
                    foot = self._compute_circular_arc(base_pos, swing_phase, side='right')
                else:
                    swing_phase = (phase - self.blend_width) / (self.right_swing_end - 2 * self.blend_width)
                    swing_phase = np.clip(swing_phase, 0.0, 1.0)
                    foot = self._compute_circular_arc(base_pos, swing_phase, side='right')
            elif phase < self.right_stance_start:
                # Transition phase: descend toward ground
                transition_phase = (phase - self.right_swing_end) / (self.right_stance_start - self.right_swing_end)
                foot = self._compute_descent_trajectory(base_pos, transition_phase, side='right')
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
        Moderate flexion with minimal lateral extension.
        """
        foot = base_pos.copy()
        
        # Minimal lateral extension based on side
        lateral_sign = -1.0 if side == 'left' else 1.0
        foot[1] += lateral_sign * self.stance_extension
        
        # Moderate downward extension for flexed stance
        foot[2] += self.stance_vertical_offset
        
        return foot

    def _compute_circular_arc(self, base_pos, swing_phase, side):
        """
        Compute foot position during overhead arc swing using small circular trajectory.
        Reduced arc radius and angle range to minimize joint extension.
        """
        foot = base_pos.copy()
        
        # Reduced arc angle range: -45 to +45 degrees
        half_range = self.arc_angle_range / 2.0
        
        if side == 'left':
            # Left legs: arc from left to right
            arc_angle = -half_range + self.arc_angle_range * swing_phase
        else:
            # Right legs: arc from right to left
            arc_angle = half_range - self.arc_angle_range * swing_phase
        
        # Circular arc trajectory in body frame (y-z plane) with small radius
        foot[1] = base_pos[1] + self.leg_arc_radius * np.cos(arc_angle)
        foot[2] = base_pos[2] + self.leg_arc_radius * np.sin(arc_angle) + self.vertical_offset_swing
        
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
        
        start_y = base_pos[1] + self.leg_arc_radius * np.cos(end_arc_angle)
        start_z = base_pos[2] + self.leg_arc_radius * np.sin(end_arc_angle) + self.vertical_offset_swing
        
        # Target stance position
        end_y = base_pos[1] - self.stance_extension
        end_z = base_pos[2] + self.stance_vertical_offset
        
        # Smooth interpolation with cubic easing
        alpha = 3 * return_phase**2 - 2 * return_phase**3
        foot[1] = start_y + (end_y - start_y) * alpha
        foot[2] = start_z + (end_z - start_z) * alpha
        
        return foot

    def _compute_descent_trajectory(self, base_pos, transition_phase, side):
        """
        Compute foot position during descent transition (right legs before stance).
        Smooth transition from overhead position to ground contact.
        """
        foot = base_pos.copy()
        
        # End of arc position
        half_range = self.arc_angle_range / 2.0
        end_arc_angle = -half_range  # Left side for right legs
        
        start_y = base_pos[1] + self.leg_arc_radius * np.cos(end_arc_angle)
        start_z = base_pos[2] + self.leg_arc_radius * np.sin(end_arc_angle) + self.vertical_offset_swing
        
        # Target stance position
        end_y = base_pos[1] + self.stance_extension
        end_z = base_pos[2] + self.stance_vertical_offset
        
        # Smooth interpolation with gentle descent curve
        alpha = 3 * transition_phase**2 - 2 * transition_phase**3
        foot[1] = start_y + (end_y - start_y) * alpha
        foot[2] = start_z + (end_z - start_z) * alpha
        
        return foot