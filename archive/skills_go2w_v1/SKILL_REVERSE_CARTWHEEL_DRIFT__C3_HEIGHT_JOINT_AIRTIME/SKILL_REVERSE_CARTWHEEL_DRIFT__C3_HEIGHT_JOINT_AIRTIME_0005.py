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
    - Legs trace very small circular arcs in body frame during swing phases
    - Deep stance flexion to provide extension reserve for transitions
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Cartwheel motion parameters
        self.roll_rate = 2.0 * np.pi * self.freq  # 360 degrees per cycle
        self.backward_velocity = -0.5  # Reduced for stability
        
        # Leg trajectory parameters - very conservative to respect joint limits
        self.leg_arc_radius = 0.08  # Very small radius
        self.stance_extension = 0.06  # Minimal lateral extension
        self.vertical_offset_swing = 0.0  # No vertical offset
        self.stance_vertical_offset_base = -0.15  # Deep flexion for extension reserve
        
        # Arc angle limits - 90 degrees total sweep
        self.arc_angle_range = np.pi / 2.0
        
        # Vertical velocity adjustments
        self.vz_ascent = 0.06
        self.vz_descent = -0.06
        
        # Phase timing for contact transitions
        self.left_stance_end = 0.2
        self.left_swing_end = 0.8
        self.right_swing_start = 0.0
        self.right_swing_end = 0.7
        self.right_stance_start = 0.8
        
        # Wider blend zones for gradual transitions
        self.blend_width = 0.15
        
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
        """
        # Reduced backward velocity
        vx = self.backward_velocity
        
        # Phase-based vertical velocity adjustments
        if phase < 0.25:
            vz = self.vz_ascent
        elif 0.5 <= phase < 0.75:
            vz = self.vz_descent
        else:
            vz = 0.0
        
        # Constant roll rate
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
        Compute foot position in body frame with phase-dependent stance depth modulation.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg side
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        if is_left:
            # Left leg group: FL, RL
            if phase < self.left_stance_end:
                # Stance phase with pre-transition flexion
                foot = self._compute_stance_position(base_pos, side='left', phase=phase, 
                                                    transition_phase=self.left_stance_end)
                
                # Blend at exit
                if phase >= self.left_stance_end - self.blend_width:
                    blend_phase = (phase - (self.left_stance_end - self.blend_width)) / self.blend_width
                    arc_foot = self._compute_circular_arc(base_pos, 0.0, side='left')
                    foot = self._smooth_blend(foot, arc_foot, blend_phase)
                    
            elif phase < self.left_swing_end:
                # Swing phase: circular arc
                # Calculate swing phase within the non-blend region
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
                # Return phase
                return_phase = (phase - self.left_swing_end) / (1.0 - self.left_swing_end)
                foot = self._compute_return_trajectory(base_pos, return_phase, side='left')
        
        elif is_right:
            # Right leg group: FR, RR
            if phase < self.right_swing_end:
                # Swing phase: circular arc
                blend_end = self.right_swing_end - self.blend_width
                
                if phase < self.blend_width:
                    # Blend from stance to arc
                    blend_phase = phase / self.blend_width
                    stance_foot = self._compute_stance_position(base_pos, side='right', phase=1.0, 
                                                               transition_phase=1.0)
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
                # Transition phase
                transition_phase = (phase - self.right_swing_end) / (self.right_stance_start - self.right_swing_end)
                foot = self._compute_descent_trajectory(base_pos, transition_phase, side='right')
                
            else:
                # Stance phase with pre-transition flexion
                foot = self._compute_stance_position(base_pos, side='right', phase=phase, 
                                                    transition_phase=1.0)
        else:
            foot = base_pos
        
        return foot

    def _smooth_blend(self, pos1, pos2, alpha):
        """Smooth interpolation using smoothstep."""
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3
        return pos1 + (pos2 - pos1) * alpha_smooth

    def _compute_stance_position(self, base_pos, side, phase, transition_phase):
        """
        Compute stance position with phase-dependent depth modulation.
        Deeper flexion before transitions to provide extension reserve.
        """
        foot = base_pos.copy()
        
        # Minimal lateral extension
        lateral_sign = -1.0 if side == 'left' else 1.0
        foot[1] += lateral_sign * self.stance_extension
        
        # Phase-dependent vertical offset - deeper near transitions
        if side == 'left':
            # Deepen stance as approaching phase 0.2
            if phase < transition_phase:
                proximity = (transition_phase - phase) / transition_phase
                # More flexion as we approach transition
                extra_flexion = 0.05 * (1.0 - proximity) if proximity < 0.3 else 0.0
            else:
                extra_flexion = 0.0
        else:
            # Right legs: deepen near phase 1.0
            proximity = 1.0 - phase
            extra_flexion = 0.05 * (1.0 - proximity) if proximity < 0.3 else 0.0
        
        foot[2] += self.stance_vertical_offset_base - extra_flexion
        
        return foot

    def _compute_circular_arc(self, base_pos, swing_phase, side):
        """
        Compute foot position during overhead arc with very small radius.
        """
        foot = base_pos.copy()
        
        # Reduced arc angle range: -45 to +45 degrees
        half_range = self.arc_angle_range / 2.0
        
        if side == 'left':
            arc_angle = -half_range + self.arc_angle_range * swing_phase
        else:
            arc_angle = half_range - self.arc_angle_range * swing_phase
        
        # Very small circular arc with no vertical offset
        foot[1] = base_pos[1] + self.leg_arc_radius * np.cos(arc_angle)
        foot[2] = base_pos[2] + self.leg_arc_radius * np.sin(arc_angle) + self.vertical_offset_swing
        
        return foot

    def _compute_return_trajectory(self, base_pos, return_phase, side):
        """
        Smooth transition from arc end to stance position.
        """
        foot = base_pos.copy()
        
        # End of arc position
        half_range = self.arc_angle_range / 2.0
        end_arc_angle = half_range
        
        start_y = base_pos[1] + self.leg_arc_radius * np.cos(end_arc_angle)
        start_z = base_pos[2] + self.leg_arc_radius * np.sin(end_arc_angle) + self.vertical_offset_swing
        
        # Target stance position with deep flexion
        end_y = base_pos[1] - self.stance_extension
        end_z = base_pos[2] + self.stance_vertical_offset_base
        
        # Smooth interpolation
        alpha = 3 * return_phase**2 - 2 * return_phase**3
        foot[1] = start_y + (end_y - start_y) * alpha
        foot[2] = start_z + (end_z - start_z) * alpha
        
        return foot

    def _compute_descent_trajectory(self, base_pos, transition_phase, side):
        """
        Smooth transition from arc end to stance position for right legs.
        """
        foot = base_pos.copy()
        
        # End of arc position
        half_range = self.arc_angle_range / 2.0
        end_arc_angle = -half_range
        
        start_y = base_pos[1] + self.leg_arc_radius * np.cos(end_arc_angle)
        start_z = base_pos[2] + self.leg_arc_radius * np.sin(end_arc_angle) + self.vertical_offset_swing
        
        # Target stance position with deep flexion
        end_y = base_pos[1] + self.stance_extension
        end_z = base_pos[2] + self.stance_vertical_offset_base
        
        # Smooth interpolation
        alpha = 3 * transition_phase**2 - 2 * transition_phase**3
        foot[1] = start_y + (end_y - start_y) * alpha
        foot[2] = start_z + (end_z - start_z) * alpha
        
        return foot