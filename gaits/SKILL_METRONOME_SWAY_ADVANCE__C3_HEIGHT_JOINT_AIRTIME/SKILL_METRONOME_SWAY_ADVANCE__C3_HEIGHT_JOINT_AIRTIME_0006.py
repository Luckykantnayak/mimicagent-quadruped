from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_METRONOME_SWAY_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Metronome sway advance gait with large-amplitude lateral base rolling.
    
    Motion characteristics:
    - All four feet maintain ground contact throughout the cycle
    - Base rolls laterally with smooth sinusoidal pattern
    - Forward surges occur during neutral roll transitions
    - Legs compress/extend asymmetrically to accommodate lateral sway
    - Optimized parameters to respect joint limits while maintaining contact
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for stable sway motion
        
        # Base foot positions (BODY frame) - use original positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - optimized to avoid joint violations
        self.max_roll_angle = np.deg2rad(12.0)  # Reduced to 12° for joint safety
        self.leg_compression_range = 0.16  # Moderate range (±0.08m per leg)
        
        # Forward surge parameters
        self.surge_velocity = 0.7  # Forward velocity during neutral phases
        self.drift_velocity = 0.1  # Minimal forward drift during sway phases
        
        # Lateral motion parameters
        self.lateral_velocity_amp = 0.22  # Moderate lateral velocity
        
        # Vertical motion parameters - minimal oscillation, no downward bias
        self.vertical_velocity_amp = 0.015  # Very small amplitude
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with lateral rolling and forward surges.
        
        Phase structure:
        - [0.0, 0.25]: Right sway - roll right, lateral right shift, minimal forward
        - [0.25, 0.5]: Neutral transition 1 - unroll to neutral, forward surge
        - [0.5, 0.75]: Left sway - roll left, lateral left shift, minimal forward
        - [0.75, 1.0]: Neutral transition 2 - unroll to neutral, forward surge
        """
        
        # Compute target roll angle using smooth sinusoidal profile
        # Peak right (+12°) at phase 0.25, peak left (-12°) at phase 0.75
        target_roll = self.max_roll_angle * np.sin(2 * np.pi * phase)
        
        # Compute roll rate to achieve target roll angle
        roll_rate = self.max_roll_angle * 2 * np.pi * self.freq * np.cos(2 * np.pi * phase)
        
        # Forward velocity: high during neutral transitions, low during sway
        if 0.25 <= phase < 0.5 or 0.75 <= phase < 1.0:
            # Neutral transition phases - forward surge
            vx = self.surge_velocity
        else:
            # Sway phases - minimal forward drift
            vx = self.drift_velocity
        
        # Lateral velocity: rightward during right sway, leftward during left sway
        vy = self.lateral_velocity_amp * np.cos(2 * np.pi * phase)
        
        # Vertical velocity: gentle oscillation with no bias
        # Two cycles per gait cycle for smooth motion
        vz = self.vertical_velocity_amp * np.sin(4 * np.pi * phase)
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame with compression/extension pattern.
        Uses phase lead and roll compensation to reduce joint stress.
        
        Right legs (FR, RR):
        - Compress during right sway [0.0, 0.25]
        - Extend to neutral during transition [0.25, 0.5]
        - Extend during left sway [0.5, 0.75]
        - Return to neutral during transition [0.75, 1.0]
        
        Left legs (FL, RL):
        - Extend during right sway [0.0, 0.25]
        - Return to neutral during transition [0.25, 0.5]
        - Compress during left sway [0.5, 0.75]
        - Extend to neutral during transition [0.75, 1.0]
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a right-side or left-side leg
        is_right_leg = leg_name in ['FR', 'RR']
        
        # Compute current roll angle for compensation
        current_roll = self.max_roll_angle * np.sin(2 * np.pi * phase)
        
        # Apply small phase lead so legs pre-compress before roll peaks
        phase_lead = 0.025
        phase_shifted = phase + phase_lead
        if phase_shifted > 1.0:
            phase_shifted -= 1.0
        
        # Compute vertical offset with roll compensation
        # Scale compression range to account for body-frame to world-frame transformation
        roll_compensation = 1.0 / max(np.cos(abs(current_roll)), 0.95)  # Limit max scaling to 1.05
        effective_range = self.leg_compression_range * roll_compensation
        
        if is_right_leg:
            # Right legs: compress when phase near 0.25, extend when phase near 0.75
            vertical_offset = -effective_range * 0.5 * np.sin(2 * np.pi * phase_shifted)
        else:
            # Left legs: extend when phase near 0.25, compress when phase near 0.75
            vertical_offset = effective_range * 0.5 * np.sin(2 * np.pi * phase_shifted)
        
        # Apply vertical offset
        foot[2] += vertical_offset
        
        # Minimal lateral adjustment to accommodate roll geometry
        lateral_adjustment = 0.025 * np.sin(2 * np.pi * phase)
        if is_right_leg:
            foot[1] -= lateral_adjustment  # Move slightly inward during right roll
        else:
            foot[1] += lateral_adjustment  # Move slightly outward during right roll
        
        return foot