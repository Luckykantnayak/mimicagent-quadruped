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
        self.roll_rate_amplitude = 0.12  # Banking roll for smooth motion
        
        # Leg compression/extension parameters - recalibrated for roll compensation
        self.compression_amount = 0.003  # Significantly reduced to prevent ground penetration when combined with roll
        self.extension_amount = 0.002  # Proportionally reduced
        self.lateral_shift = 0.045  # Increased lateral shift to maintain banking effect visually
        
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

    def get_predicted_roll_magnitude(self, phase):
        """
        Compute predicted maximum roll magnitude for the current phase region.
        Uses phase-based prediction rather than instantaneous roll angle.
        """
        # Roll follows: roll_rate_amplitude * (-sin(2*pi*phase))
        # Integrated over time: roll(t) ≈ (roll_rate_amplitude / (2*pi*freq)) * cos(2*pi*phase)
        # Maximum roll magnitude
        max_roll = self.roll_rate_amplitude / (2 * np.pi * self.freq)
        
        # Compute the magnitude of roll at current phase
        # Roll is maximum at phase 0.0 and 0.5, minimum at 0.25 and 0.75
        roll_phase_factor = abs(np.cos(2 * np.pi * phase))
        
        predicted_roll = max_roll * roll_phase_factor
        return predicted_roll

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with differential compression.
        
        Inside legs (relative to turn direction) compress downward.
        Outside legs extend upward to create banking effect.
        All feet maintain ground contact.
        
        Uses phase-based predictive roll compensation synchronized with compression phase.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left or right leg
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Synchronized compression phase (no offset to ensure accurate roll prediction)
        compression_phase = phase
        
        # Get predicted roll magnitude for compensation
        predicted_roll = self.get_predicted_roll_magnitude(compression_phase)
        
        # Compute aggressive roll compensation: reduce compression significantly when roll is present
        # At maximum roll (~0.064 rad), we want to reduce compression by ~80%
        max_expected_roll = 0.065
        roll_compensation = 1.0 - min(predicted_roll / max_expected_roll, 0.8)
        
        # Compute compression/extension with smooth transitions and phase-predictive roll compensation
        if compression_phase < 0.25:
            # Right turn compression phase
            turn_phase = compression_phase / 0.25  # 0 to 1
            # Smooth step function (3t^2 - 2t^3)
            smooth_factor = 3 * turn_phase**2 - 2 * turn_phase**3
            
            if is_right_leg:
                z_offset = -self.compression_amount * smooth_factor * roll_compensation
                y_offset = -self.lateral_shift * smooth_factor
            else:
                z_offset = self.extension_amount * smooth_factor * roll_compensation
                y_offset = self.lateral_shift * smooth_factor
                
        elif compression_phase < 0.5:
            # Center crossing 1: smooth transition with extended blend period
            transition_phase = (compression_phase - 0.25) / 0.25  # 0 to 1
            # Use smoothstep for more gradual transition
            blend = 1.0 - (3 * transition_phase**2 - 2 * transition_phase**3)
            
            if is_right_leg:
                z_offset = -self.compression_amount * blend * roll_compensation
                y_offset = -self.lateral_shift * blend
            else:
                z_offset = self.extension_amount * blend * roll_compensation
                y_offset = self.lateral_shift * blend
                
        elif compression_phase < 0.75:
            # Left turn compression phase
            turn_phase = (compression_phase - 0.5) / 0.25  # 0 to 1
            smooth_factor = 3 * turn_phase**2 - 2 * turn_phase**3
            
            if is_left_leg:
                z_offset = -self.compression_amount * smooth_factor * roll_compensation
                y_offset = self.lateral_shift * smooth_factor
            else:
                z_offset = self.extension_amount * smooth_factor * roll_compensation
                y_offset = -self.lateral_shift * smooth_factor
                
        else:
            # Center crossing 2: smooth transition with extended blend period
            transition_phase = (compression_phase - 0.75) / 0.25  # 0 to 1
            blend = 1.0 - (3 * transition_phase**2 - 2 * transition_phase**3)
            
            if is_left_leg:
                z_offset = -self.compression_amount * blend * roll_compensation
                y_offset = self.lateral_shift * blend
            else:
                z_offset = self.extension_amount * blend * roll_compensation
                y_offset = -self.lateral_shift * blend
        
        # Apply offsets with smooth interpolation
        foot[1] += y_offset  # Lateral shift in body frame
        foot[2] += z_offset  # Vertical compression/extension with phase-predictive roll compensation
        
        return foot