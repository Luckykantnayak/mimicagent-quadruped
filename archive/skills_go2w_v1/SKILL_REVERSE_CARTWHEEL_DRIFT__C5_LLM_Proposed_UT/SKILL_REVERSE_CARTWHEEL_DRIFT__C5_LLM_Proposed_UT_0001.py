from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERSE_CARTWHEEL_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Reverse cartwheel drift: continuous backward drifting motion combined with 
    a full 360-degree roll rotation per cycle.
    
    - Base maintains constant backward velocity (negative x)
    - Base executes continuous positive roll rate (360 degrees per cycle)
    - Right legs (FR, RR) extend overhead during [0, 0.5]
    - Left legs (FL, RL) extend overhead during [0.5, 1.0]
    - Alternating lateral support pattern creates cartwheel locomotion
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Cycle frequency (Hz) - controls cartwheel speed
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.backward_velocity = -0.8  # Sustained backward drift (m/s)
        self.roll_rate = 2 * np.pi * self.freq  # 360 degrees per cycle (rad/s)
        
        # Leg extension parameters
        self.extension_amplitude = 0.25  # Maximum overhead extension distance (m)
        self.lateral_offset = 0.15  # Lateral displacement during stance (m)
        self.vertical_stance = -0.05  # Vertical offset during stance (m)
        
    def update_base_motion(self, phase, dt):
        """
        Update base with constant backward velocity and continuous positive roll rate.
        """
        # Constant backward velocity throughout cycle
        vx = self.backward_velocity
        
        # Constant positive roll rate for 360-degree rotation per cycle
        roll_rate = self.roll_rate
        
        self.vel_world = np.array([vx, 0.0, 0.0])
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
        Compute foot position in body frame based on cartwheel phase.
        
        Right legs (FR, RR): overhead during [0, 0.5], stance during [0.5, 1.0]
        Left legs (FL, RL): stance during [0, 0.5], overhead during [0.5, 1.0]
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is on right or left side
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        if is_right_leg:
            # Right legs: swing [0, 0.5], stance [0.5, 1.0]
            if phase < 0.5:
                # Overhead swing phase
                foot = self._compute_overhead_trajectory(base_pos, phase, 0.0, 0.5, is_right=True)
            else:
                # Stance phase
                foot = self._compute_stance_trajectory(base_pos, phase, 0.5, 1.0, is_right=True)
        else:
            # Left legs: stance [0, 0.5], swing [0.5, 1.0]
            if phase < 0.5:
                # Stance phase
                foot = self._compute_stance_trajectory(base_pos, phase, 0.0, 0.5, is_right=False)
            else:
                # Overhead swing phase
                foot = self._compute_overhead_trajectory(base_pos, phase, 0.5, 1.0, is_right=False)
        
        return foot
    
    def _compute_overhead_trajectory(self, base_pos, phase, phase_start, phase_end, is_right):
        """
        Compute overhead circular arc trajectory for swing phase.
        Leg extends away from body center in lateral-upward direction.
        """
        # Normalize phase within swing interval
        local_phase = (phase - phase_start) / (phase_end - phase_start)
        
        # Circular arc from stance to overhead and back
        # Arc angle: 0 -> pi (semicircle)
        arc_angle = np.pi * local_phase
        
        foot = base_pos.copy()
        
        # Lateral direction: positive y for right legs, negative y for left legs
        lateral_sign = 1.0 if is_right else -1.0
        
        # Circular trajectory in y-z plane (body frame)
        # Start from stance position, arc overhead, return to stance
        radius = self.extension_amplitude
        
        # Y-component: lateral extension (peaks at mid-swing)
        foot[1] = base_pos[1] + lateral_sign * radius * np.sin(arc_angle)
        
        # Z-component: vertical extension (peaks at mid-swing)
        # Offset to ensure smooth transition from/to stance height
        foot[2] = base_pos[2] + radius * (1.0 - np.cos(arc_angle))
        
        # X-component: slight forward reach during overhead phase
        foot[0] = base_pos[0] + 0.05 * np.sin(arc_angle)
        
        return foot
    
    def _compute_stance_trajectory(self, base_pos, phase, phase_start, phase_end, is_right):
        """
        Compute stance trajectory with lateral offset and backward propulsion.
        Leg maintains ground contact and provides support.
        """
        # Normalize phase within stance interval
        local_phase = (phase - phase_start) / (phase_end - phase_start)
        
        foot = base_pos.copy()
        
        # Lateral direction: positive y for right legs, negative y for left legs
        lateral_sign = 1.0 if is_right else -1.0
        
        # Lateral offset for stable stance
        foot[1] = base_pos[1] + lateral_sign * self.lateral_offset
        
        # Vertical stance position (slightly below base)
        foot[2] = base_pos[2] + self.vertical_stance
        
        # Backward sweep during stance to generate propulsion
        # Foot moves backward relative to body as base drifts backward
        sweep_amplitude = 0.08
        foot[0] = base_pos[0] - sweep_amplitude * (local_phase - 0.5)
        
        return foot