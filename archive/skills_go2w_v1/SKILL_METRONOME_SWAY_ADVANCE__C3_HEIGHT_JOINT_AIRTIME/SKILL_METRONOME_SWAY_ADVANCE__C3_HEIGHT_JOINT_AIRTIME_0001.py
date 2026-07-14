from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_METRONOME_SWAY_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Metronome sway advance gait: forward locomotion through large-amplitude lateral base roll oscillations (±30°).
    
    - All four feet remain in contact throughout (contact-rich gait)
    - Base rolls side-to-side with period determined by freq
    - Forward surges occur during neutral roll transitions
    - Legs compress/extend asymmetrically as lateral pairs to accommodate roll
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Hz, full sway cycle frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Leg group identification (lateral pairs)
        self.left_legs = [name for name in leg_names if name.startswith('FL') or name.startswith('RL')]
        self.right_legs = [name for name in leg_names if name.startswith('FR') or name.startswith('RR')]
        
        # Motion parameters
        self.peak_roll_angle = np.deg2rad(30.0)  # Target peak roll amplitude
        self.surge_vx = 0.8  # Forward velocity during surge phases (m/s)
        self.drift_vx = 0.1  # Minimal forward velocity during sway phases (m/s)
        self.lateral_vy_amp = 0.15  # Lateral velocity amplitude (m/s)
        
        # Leg motion parameters
        self.leg_compression_range = 0.06  # Vertical compression/extension range (m)
        self.leg_lateral_shift = 0.02  # Small lateral foot adjustment during sway (m)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with lateral roll oscillations and forward surges.
        
        Phase breakdown:
        [0.0, 0.25]: right sway - roll right, minimal forward, rightward drift
        [0.25, 0.5]: neutral surge 1 - unwind roll, forward surge, lateral reversal
        [0.5, 0.75]: left sway - roll left, minimal forward, leftward drift
        [0.75, 1.0]: neutral surge 2 - unwind roll, forward surge, lateral reversal
        """
        
        # Determine sub-phase and local progress
        if phase < 0.25:
            # Right sway phase
            sub_phase_progress = phase / 0.25
            target_roll = self.peak_roll_angle
            roll_rate_sign = 1.0
            vx = self.drift_vx
            vy = self.lateral_vy_amp * np.sin(np.pi * sub_phase_progress)
            vz = -0.02  # Slight downward during sway
            
        elif phase < 0.5:
            # Neutral surge 1
            sub_phase_progress = (phase - 0.25) / 0.25
            target_roll = self.peak_roll_angle * (1.0 - sub_phase_progress)
            roll_rate_sign = -1.0
            vx = self.surge_vx
            vy = self.lateral_vy_amp * np.cos(np.pi * (sub_phase_progress + 0.5))
            vz = 0.02  # Slight upward during neutral transition
            
        elif phase < 0.75:
            # Left sway phase
            sub_phase_progress = (phase - 0.5) / 0.25
            target_roll = -self.peak_roll_angle
            roll_rate_sign = -1.0
            vx = self.drift_vx
            vy = -self.lateral_vy_amp * np.sin(np.pi * sub_phase_progress)
            vz = -0.02  # Slight downward during sway
            
        else:
            # Neutral surge 2
            sub_phase_progress = (phase - 0.75) / 0.25
            target_roll = -self.peak_roll_angle * (1.0 - sub_phase_progress)
            roll_rate_sign = 1.0
            vx = self.surge_vx
            vy = -self.lateral_vy_amp * np.cos(np.pi * (sub_phase_progress + 0.5))
            vz = 0.02  # Slight upward during neutral transition
        
        # Compute smooth roll rate based on sinusoidal profile
        # Roll rate modulated to achieve target roll angles smoothly
        omega_cycle = 2.0 * np.pi * self.freq
        roll_rate = roll_rate_sign * self.peak_roll_angle * omega_cycle * 0.8
        
        # Smooth transitions at phase boundaries using cosine taper
        if phase < 0.05 or phase > 0.95:
            taper = 0.5 * (1.0 - np.cos(np.pi * min(phase / 0.05, (1.0 - phase) / 0.05)))
            roll_rate *= taper
        
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
        Compute foot position in body frame with vertical compression/extension 
        to accommodate base roll oscillations.
        
        Left legs (FL, RL): extend during right sway, compress during left sway
        Right legs (FR, RR): compress during right sway, extend during left sway
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name in self.left_legs
        is_right_leg = leg_name in self.right_legs
        
        # Compute compression/extension factor based on phase
        # Positive = extension (downward in body frame), Negative = compression (upward)
        if phase < 0.25:
            # Right sway phase
            sub_progress = phase / 0.25
            compression_profile = np.sin(np.pi * sub_progress)
            if is_left_leg:
                delta_z = -self.leg_compression_range * compression_profile  # Extend
                delta_y = -self.leg_lateral_shift * compression_profile  # Shift left
            else:
                delta_z = self.leg_compression_range * compression_profile  # Compress
                delta_y = self.leg_lateral_shift * compression_profile  # Shift right
                
        elif phase < 0.5:
            # Neutral surge 1
            sub_progress = (phase - 0.25) / 0.25
            compression_profile = np.cos(np.pi * sub_progress * 0.5)
            if is_left_leg:
                delta_z = -self.leg_compression_range * compression_profile
                delta_y = -self.leg_lateral_shift * compression_profile
            else:
                delta_z = self.leg_compression_range * compression_profile
                delta_y = self.leg_lateral_shift * compression_profile
                
        elif phase < 0.75:
            # Left sway phase
            sub_progress = (phase - 0.5) / 0.25
            compression_profile = np.sin(np.pi * sub_progress)
            if is_left_leg:
                delta_z = self.leg_compression_range * compression_profile  # Compress
                delta_y = self.leg_lateral_shift * compression_profile  # Shift left
            else:
                delta_z = -self.leg_compression_range * compression_profile  # Extend
                delta_y = -self.leg_lateral_shift * compression_profile  # Shift right
                
        else:
            # Neutral surge 2
            sub_progress = (phase - 0.75) / 0.25
            compression_profile = np.cos(np.pi * sub_progress * 0.5)
            if is_left_leg:
                delta_z = self.leg_compression_range * compression_profile
                delta_y = self.leg_lateral_shift * compression_profile
            else:
                delta_z = -self.leg_compression_range * compression_profile
                delta_y = -self.leg_lateral_shift * compression_profile
        
        # Apply vertical compression/extension
        foot[2] += delta_z
        
        # Apply small lateral shift to maintain stability during roll
        foot[1] += delta_y
        
        return foot