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
    - Legs trace circular arcs in body frame during swing phases
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
        
        # Leg trajectory parameters
        self.leg_arc_radius = 0.35  # Radius of overhead circular arc
        self.stance_extension = 0.15  # Lateral extension during stance
        self.vertical_offset_swing = 0.1  # Additional height during overhead arc
        
        # Vertical velocity adjustments for transitions
        self.vz_ascent = 0.15  # Upward velocity during initial roll
        self.vz_descent = -0.15  # Downward velocity during descent phase
        
        # Phase timing for contact transitions
        self.left_stance_end = 0.2
        self.left_swing_end = 0.8
        self.right_swing_start = 0.0
        self.right_swing_end = 0.7
        self.right_stance_start = 0.8
        
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
        Vertical velocity adjustments during ascent and descent phases.
        """
        # Constant backward velocity
        vx = self.backward_velocity
        
        # Vertical velocity adjustments based on phase
        if phase < 0.25:
            # Ascent phase: slight upward velocity
            vz = self.vz_ascent
        elif 0.5 <= phase < 0.75:
            # Descent phase: slight downward velocity
            vz = self.vz_descent
        else:
            # Inverted and recovery phases: neutral vertical velocity
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
                # Stance phase: positioned laterally left and down
                foot = self._compute_stance_position(base_pos, side='left', phase=phase, phase_end=self.left_stance_end)
            elif phase < self.left_swing_end:
                # Swing phase: overhead arc
                swing_phase = (phase - self.left_stance_end) / (self.left_swing_end - self.left_stance_end)
                foot = self._compute_overhead_arc(base_pos, swing_phase, side='left')
            else:
                # Return phase: descend and reposition
                return_phase = (phase - self.left_swing_end) / (1.0 - self.left_swing_end)
                foot = self._compute_return_trajectory(base_pos, return_phase, side='left')
        
        elif is_right:
            # Right leg group: FR, RR
            if phase < self.right_swing_end:
                # Swing phase: overhead arc
                swing_phase = phase / self.right_swing_end
                foot = self._compute_overhead_arc(base_pos, swing_phase, side='right')
            elif phase < self.right_stance_start:
                # Transition phase: descend toward ground
                transition_phase = (phase - self.right_swing_end) / (self.right_stance_start - self.right_swing_end)
                foot = self._compute_descent_trajectory(base_pos, transition_phase, side='right')
            else:
                # Stance phase: positioned laterally right and down
                stance_phase = (phase - self.right_stance_start) / (1.0 - self.right_stance_start)
                foot = self._compute_stance_position(base_pos, side='right', phase=stance_phase, phase_end=1.0)
        else:
            foot = base_pos
        
        return foot

    def _compute_stance_position(self, base_pos, side, phase, phase_end):
        """
        Compute foot position during stance phase.
        Foot extends laterally and maintains ground contact.
        """
        foot = base_pos.copy()
        
        # Lateral extension based on side
        lateral_sign = -1.0 if side == 'left' else 1.0
        foot[1] += lateral_sign * self.stance_extension
        
        # Slight downward extension for ground contact
        foot[2] -= 0.05
        
        return foot

    def _compute_overhead_arc(self, base_pos, swing_phase, side):
        """
        Compute foot position during overhead arc swing.
        Leg traces circular arc in body frame from one side to the other.
        """
        foot = base_pos.copy()
        
        # Arc angle: 0 to pi (180 degrees overhead sweep)
        if side == 'left':
            # Left legs: arc from left (-pi/2) to right (+pi/2)
            arc_angle = -np.pi/2 + np.pi * swing_phase
        else:
            # Right legs: arc from right (+pi/2) to left (-pi/2)
            arc_angle = np.pi/2 - np.pi * swing_phase
        
        # Circular arc trajectory in body frame (y-z plane)
        foot[1] = base_pos[1] + self.leg_arc_radius * np.cos(arc_angle)
        foot[2] = base_pos[2] + self.leg_arc_radius * np.sin(arc_angle) + self.vertical_offset_swing
        
        return foot

    def _compute_return_trajectory(self, base_pos, return_phase, side):
        """
        Compute foot position during return phase (left legs after overhead arc).
        Smooth transition from overhead position back to stance preparation.
        """
        foot = base_pos.copy()
        
        # Interpolate from end of arc to stance position
        start_angle = np.pi/2  # End of overhead arc (right side)
        end_y = base_pos[1] - self.stance_extension  # Target lateral position (left)
        end_z = base_pos[2] - 0.05  # Target vertical position (slightly down)
        
        start_y = base_pos[1] + self.leg_arc_radius * np.cos(start_angle)
        start_z = base_pos[2] + self.leg_arc_radius * np.sin(start_angle) + self.vertical_offset_swing
        
        # Smooth interpolation
        foot[1] = start_y + (end_y - start_y) * return_phase
        foot[2] = start_z + (end_z - start_z) * return_phase
        
        return foot

    def _compute_descent_trajectory(self, base_pos, transition_phase, side):
        """
        Compute foot position during descent transition (right legs before stance).
        Smooth transition from overhead position to ground contact.
        """
        foot = base_pos.copy()
        
        # Interpolate from end of arc to stance position
        start_angle = -np.pi/2  # End of overhead arc (left side)
        end_y = base_pos[1] + self.stance_extension  # Target lateral position (right)
        end_z = base_pos[2] - 0.05  # Target vertical position (slightly down)
        
        start_y = base_pos[1] + self.leg_arc_radius * np.cos(start_angle)
        start_z = base_pos[2] + self.leg_arc_radius * np.sin(start_angle) + self.vertical_offset_swing
        
        # Smooth interpolation with slight parabolic descent for natural landing
        foot[1] = start_y + (end_y - start_y) * transition_phase
        foot[2] = start_z + (end_z - start_z) * (transition_phase ** 0.8)
        
        return foot