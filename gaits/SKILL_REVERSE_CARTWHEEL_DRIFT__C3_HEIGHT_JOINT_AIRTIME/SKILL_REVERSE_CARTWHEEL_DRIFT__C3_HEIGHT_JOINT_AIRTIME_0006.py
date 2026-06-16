from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERSE_CARTWHEEL_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Reverse cartwheel drift: continuous backward translation with 360-degree roll per cycle.
    
    Uses roll-angle-aware foot positioning to maintain kinematic feasibility throughout rotation.
    Foot positions are modulated based on body orientation to prevent knee hyperextension.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Cartwheel motion parameters
        self.roll_rate = 2.0 * np.pi * self.freq  # 360 degrees per cycle
        self.backward_velocity = -0.5
        
        # Leg trajectory parameters - with roll-angle compensation
        self.base_arc_radius = 0.06  # Small base radius
        self.stance_extension = 0.0  # No lateral extension - rely on roll for cartwheel effect
        self.vertical_offset_swing = 0.0
        
        # Stance depth parameters - modulated by roll angle
        self.base_stance_depth = -0.10  # Moderate base flexion
        self.min_stance_depth = -0.03  # Minimal flexion when inverted
        
        # Arc angle limits
        self.arc_angle_range = np.pi / 2.0  # 90 degrees
        
        # Vertical velocity adjustments
        self.vz_ascent = 0.05
        self.vz_descent = -0.05
        
        # Phase timing
        self.left_stance_end = 0.2
        self.left_swing_end = 0.8
        self.right_swing_end = 0.7
        self.right_stance_start = 0.8
        
        # Blend zones
        self.blend_width = 0.12
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Current roll angle tracking
        self.current_roll = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base with continuous backward velocity and constant roll rate.
        Track current roll angle for orientation-dependent foot positioning.
        """
        vx = self.backward_velocity
        
        # Phase-based vertical velocity
        if phase < 0.25:
            vz = self.vz_ascent
        elif 0.5 <= phase < 0.75:
            vz = self.vz_descent
        else:
            vz = 0.0
        
        roll_rate = self.roll_rate
        
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
        
        # Extract current roll angle from quaternion for orientation-aware positioning
        self.current_roll = self._extract_roll_from_quat(self.root_quat)

    def _extract_roll_from_quat(self, quat):
        """
        Extract roll angle from quaternion.
        quat = [w, x, y, z]
        Roll is rotation about x-axis.
        """
        w, x, y, z = quat
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        return roll

    def _get_roll_compensation_factor(self):
        """
        Compute compensation factor based on current roll angle.
        Returns value in [0, 1] where:
        - 1.0 at upright (roll = 0 or 2π)
        - 0.0 at inverted (roll = π)
        """
        # Normalize roll to [0, 2π]
        roll_normalized = self.current_roll % (2.0 * np.pi)
        # Compute cosine-based factor
        factor = (np.cos(roll_normalized) + 1.0) / 2.0
        return factor

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position with roll-angle-aware adjustments.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        if is_left:
            if phase < self.left_stance_end:
                foot = self._compute_stance_position(base_pos, side='left')
                
                if phase >= self.left_stance_end - self.blend_width:
                    blend_phase = (phase - (self.left_stance_end - self.blend_width)) / self.blend_width
                    arc_foot = self._compute_circular_arc(base_pos, 0.0, side='left')
                    foot = self._smooth_blend(foot, arc_foot, blend_phase)
                    
            elif phase < self.left_swing_end:
                blend_start = self.left_stance_end + self.blend_width
                blend_end = self.left_swing_end - self.blend_width
                
                if phase < blend_start:
                    swing_phase = 0.0
                elif phase > blend_end:
                    swing_phase = 1.0
                else:
                    swing_phase = (phase - blend_start) / (blend_end - blend_start)
                    swing_phase = np.clip(swing_phase, 0.0, 1.0)
                    
                foot = self._compute_circular_arc(base_pos, swing_phase, side='left')
                
            else:
                return_phase = (phase - self.left_swing_end) / (1.0 - self.left_swing_end)
                foot = self._compute_return_trajectory(base_pos, return_phase, side='left')
        
        elif is_right:
            if phase < self.right_swing_end:
                blend_end = self.right_swing_end - self.blend_width
                
                if phase < self.blend_width:
                    blend_phase = phase / self.blend_width
                    stance_foot = self._compute_stance_position(base_pos, side='right')
                    arc_foot = self._compute_circular_arc(base_pos, 0.0, side='right')
                    foot = self._smooth_blend(stance_foot, arc_foot, blend_phase)
                elif phase > blend_end:
                    swing_phase = 1.0
                    foot = self._compute_circular_arc(base_pos, swing_phase, side='right')
                else:
                    swing_phase = (phase - self.blend_width) / (blend_end - self.blend_width)
                    swing_phase = np.clip(swing_phase, 0.0, 1.0)
                    foot = self._compute_circular_arc(base_pos, swing_phase, side='right')
                    
            elif phase < self.right_stance_start:
                transition_phase = (phase - self.right_swing_end) / (self.right_stance_start - self.right_swing_end)
                foot = self._compute_descent_trajectory(base_pos, transition_phase, side='right')
                
            else:
                foot = self._compute_stance_position(base_pos, side='right')
        else:
            foot = base_pos
        
        return foot

    def _smooth_blend(self, pos1, pos2, alpha):
        """Smooth interpolation using smoothstep."""
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3
        return pos1 + (pos2 - pos1) * alpha_smooth

    def _compute_stance_position(self, base_pos, side):
        """
        Compute stance position with roll-angle-dependent depth.
        Reduces flexion when body is tilted/inverted to prevent knee hyperextension.
        """
        foot = base_pos.copy()
        
        # No lateral extension - feet stay under body
        lateral_sign = -1.0 if side == 'left' else 1.0
        foot[1] += lateral_sign * self.stance_extension
        
        # Roll-compensated vertical offset
        roll_factor = self._get_roll_compensation_factor()
        # Interpolate between min depth (inverted) and base depth (upright)
        stance_depth = self.min_stance_depth + roll_factor * (self.base_stance_depth - self.min_stance_depth)
        foot[2] += stance_depth
        
        return foot

    def _compute_circular_arc(self, base_pos, swing_phase, side):
        """
        Compute arc position with roll-angle-dependent radius scaling.
        Reduces arc size when body is tilted to minimize reach requirements.
        """
        foot = base_pos.copy()
        
        half_range = self.arc_angle_range / 2.0
        
        if side == 'left':
            arc_angle = -half_range + self.arc_angle_range * swing_phase
        else:
            arc_angle = half_range - self.arc_angle_range * swing_phase
        
        # Roll-compensated arc radius
        roll_factor = self._get_roll_compensation_factor()
        # Reduce radius when tilted (roll_factor low)
        arc_radius = self.base_arc_radius * (0.5 + 0.5 * roll_factor)
        
        # Circular arc with compensated radius
        foot[1] = base_pos[1] + arc_radius * np.cos(arc_angle)
        foot[2] = base_pos[2] + arc_radius * np.sin(arc_angle) + self.vertical_offset_swing
        
        return foot

    def _compute_return_trajectory(self, base_pos, return_phase, side):
        """
        Smooth transition from arc end to stance.
        """
        foot = base_pos.copy()
        
        half_range = self.arc_angle_range / 2.0
        end_arc_angle = half_range
        
        # Arc end position with current roll compensation
        roll_factor = self._get_roll_compensation_factor()
        arc_radius = self.base_arc_radius * (0.5 + 0.5 * roll_factor)
        
        start_y = base_pos[1] + arc_radius * np.cos(end_arc_angle)
        start_z = base_pos[2] + arc_radius * np.sin(end_arc_angle) + self.vertical_offset_swing
        
        # Target stance position with roll compensation
        stance_depth = self.min_stance_depth + roll_factor * (self.base_stance_depth - self.min_stance_depth)
        end_y = base_pos[1] - self.stance_extension
        end_z = base_pos[2] + stance_depth
        
        alpha = 3 * return_phase**2 - 2 * return_phase**3
        foot[1] = start_y + (end_y - start_y) * alpha
        foot[2] = start_z + (end_z - start_z) * alpha
        
        return foot

    def _compute_descent_trajectory(self, base_pos, transition_phase, side):
        """
        Smooth transition from arc end to stance for right legs.
        """
        foot = base_pos.copy()
        
        half_range = self.arc_angle_range / 2.0
        end_arc_angle = -half_range
        
        # Arc end position with current roll compensation
        roll_factor = self._get_roll_compensation_factor()
        arc_radius = self.base_arc_radius * (0.5 + 0.5 * roll_factor)
        
        start_y = base_pos[1] + arc_radius * np.cos(end_arc_angle)
        start_z = base_pos[2] + arc_radius * np.sin(end_arc_angle) + self.vertical_offset_swing
        
        # Target stance position with roll compensation
        stance_depth = self.min_stance_depth + roll_factor * (self.base_stance_depth - self.min_stance_depth)
        end_y = base_pos[1] + self.stance_extension
        end_z = base_pos[2] + stance_depth
        
        alpha = 3 * transition_phase**2 - 2 * transition_phase**3
        foot[1] = start_y + (end_y - start_y) * alpha
        foot[2] = start_z + (end_z - start_z) * alpha
        
        return foot