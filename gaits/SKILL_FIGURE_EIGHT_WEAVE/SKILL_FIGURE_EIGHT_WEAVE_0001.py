from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_FIGURE_EIGHT_WEAVE_MotionGenerator(BaseMotionGenerator):
    """
    Figure-eight weave skill with continuous ground contact.
    
    Robot traces a figure-eight pattern by alternating tight right and left turns
    with center crossings. Uses differential leg compression (banking) to shift
    weight during curved motion.
    
    - Phase 0.0-0.25: Right loop (clockwise arc), right legs compressed
    - Phase 0.25-0.5: Center crossing (right to left), legs equalize
    - Phase 0.5-0.75: Left loop (counter-clockwise arc), left legs compressed
    - Phase 0.75-1.0: Center crossing (left to right), legs equalize
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.3  # Slower frequency for smooth figure-eight
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.vx_base = 0.4  # Constant forward velocity
        self.vy_amplitude = 0.3  # Lateral velocity amplitude for carving
        self.yaw_rate_amplitude = 1.2  # Yaw rate amplitude (rad/s)
        self.roll_rate_amplitude = 0.15  # Small banking roll
        
        # Leg compression/extension parameters
        self.compression_amount = 0.06  # Vertical compression for inside legs
        self.extension_amount = 0.04  # Vertical extension for outside legs
        self.lateral_shift = 0.02  # Lateral shift during turns
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base velocity to trace figure-eight pattern.
        
        Uses sinusoidal modulation of yaw rate and lateral velocity
        to carve alternating circular arcs connected at center crossings.
        """
        
        # Forward velocity: constant throughout
        vx = self.vx_base
        
        # Lateral velocity: sinusoidal with period matching phase
        # Positive (left) during right turn, negative (right) during left turn
        vy = self.vy_amplitude * np.sin(2 * np.pi * phase)
        
        # Yaw rate: positive for right loop, negative for left loop
        # Smoothly transitions through zero at center crossings
        yaw_rate = self.yaw_rate_amplitude * np.sin(2 * np.pi * phase)
        
        # Roll rate: small banking effect, follows yaw direction
        # Negative roll (bank right) during right turn, positive during left turn
        roll_rate = -self.roll_rate_amplitude * np.sin(2 * np.pi * phase)
        
        # Set velocities
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, yaw_rate])
        
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
        Compute foot position in body frame with differential compression.
        
        Inside legs (relative to turn direction) compress downward.
        Outside legs extend upward to create banking effect.
        All feet maintain ground contact.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left or right leg
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compute compression/extension factor based on phase
        # Right turn: phase 0.0-0.25 (right legs compress, left extend)
        # Center 1: phase 0.25-0.5 (equalize)
        # Left turn: phase 0.5-0.75 (left legs compress, right extend)
        # Center 2: phase 0.75-1.0 (equalize)
        
        if phase < 0.25:
            # Right turn: right legs inside (compress), left legs outside (extend)
            turn_phase = phase / 0.25  # 0 to 1
            if is_right_leg:
                z_offset = -self.compression_amount * np.sin(np.pi * turn_phase)
                y_offset = -self.lateral_shift * np.sin(np.pi * turn_phase)
            else:
                z_offset = self.extension_amount * np.sin(np.pi * turn_phase)
                y_offset = self.lateral_shift * np.sin(np.pi * turn_phase)
                
        elif phase < 0.5:
            # Center crossing 1: transition from right turn to left turn
            transition_phase = (phase - 0.25) / 0.25  # 0 to 1
            if is_right_leg:
                z_offset = -self.compression_amount * np.cos(np.pi * transition_phase / 2)
                y_offset = -self.lateral_shift * np.cos(np.pi * transition_phase / 2)
            else:
                z_offset = self.extension_amount * np.cos(np.pi * transition_phase / 2)
                y_offset = self.lateral_shift * np.cos(np.pi * transition_phase / 2)
                
        elif phase < 0.75:
            # Left turn: left legs inside (compress), right legs outside (extend)
            turn_phase = (phase - 0.5) / 0.25  # 0 to 1
            if is_left_leg:
                z_offset = -self.compression_amount * np.sin(np.pi * turn_phase)
                y_offset = self.lateral_shift * np.sin(np.pi * turn_phase)
            else:
                z_offset = self.extension_amount * np.sin(np.pi * turn_phase)
                y_offset = -self.lateral_shift * np.sin(np.pi * turn_phase)
                
        else:
            # Center crossing 2: transition from left turn back to right turn
            transition_phase = (phase - 0.75) / 0.25  # 0 to 1
            if is_left_leg:
                z_offset = -self.compression_amount * np.cos(np.pi * transition_phase / 2)
                y_offset = self.lateral_shift * np.cos(np.pi * transition_phase / 2)
            else:
                z_offset = self.extension_amount * np.cos(np.pi * transition_phase / 2)
                y_offset = -self.lateral_shift * np.cos(np.pi * transition_phase / 2)
        
        # Apply offsets
        foot[1] += y_offset  # Lateral shift in body frame
        foot[2] += z_offset  # Vertical compression/extension
        
        return foot