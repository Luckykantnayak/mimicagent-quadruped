from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_METRONOME_SWAY_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Metronome sway advance gait with large-amplitude lateral body roll oscillations (±30°).
    
    Four-phase cycle:
    - [0.0, 0.25]: Roll right with lateral shift
    - [0.25, 0.5]: Neutral transition with forward surge
    - [0.5, 0.75]: Roll left with lateral shift
    - [0.75, 1.0]: Neutral transition with forward surge
    
    All four feet maintain continuous ground contact throughout.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slow cycle for large amplitude roll
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.peak_roll_angle = np.deg2rad(30.0)  # ±30° roll
        self.lateral_shift_amplitude = 0.08  # Lateral displacement during roll
        self.forward_surge_speed = 0.3  # Forward velocity during neutral phases
        self.leg_compression_ratio = 0.18  # ±18% length variation
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocities based on current phase.
        
        Phase structure:
        - [0.0, 0.25]: Roll right, shift right, no forward motion
        - [0.25, 0.5]: Return to neutral roll, surge forward, center laterally
        - [0.5, 0.75]: Roll left, shift left, no forward motion
        - [0.75, 1.0]: Return to neutral roll, surge forward, center laterally
        """
        
        # Phase-dependent velocity commands
        if phase < 0.25:
            # Right roll sway phase
            phase_local = phase / 0.25
            vx = 0.0
            vy = self.lateral_shift_amplitude * 4.0 * np.sin(np.pi * phase_local)
            roll_rate = (self.peak_roll_angle * 4.0) * np.cos(2 * np.pi * phase_local * 0.5)
            
        elif phase < 0.5:
            # Neutral surge 1 phase
            phase_local = (phase - 0.25) / 0.25
            vx = self.forward_surge_speed * np.sin(np.pi * phase_local)
            vy = -self.lateral_shift_amplitude * 4.0 * np.sin(np.pi * phase_local)
            roll_rate = -(self.peak_roll_angle * 4.0) * np.cos(2 * np.pi * (phase_local + 0.5) * 0.5)
            
        elif phase < 0.75:
            # Left roll sway phase
            phase_local = (phase - 0.5) / 0.25
            vx = 0.0
            vy = -self.lateral_shift_amplitude * 4.0 * np.sin(np.pi * phase_local)
            roll_rate = -(self.peak_roll_angle * 4.0) * np.cos(2 * np.pi * phase_local * 0.5)
            
        else:
            # Neutral surge 2 phase
            phase_local = (phase - 0.75) / 0.25
            vx = self.forward_surge_speed * np.sin(np.pi * phase_local)
            vy = self.lateral_shift_amplitude * 4.0 * np.sin(np.pi * phase_local)
            roll_rate = (self.peak_roll_angle * 4.0) * np.cos(2 * np.pi * (phase_local + 0.5) * 0.5)
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, 0.0])
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
        Compute foot position in body frame with leg compression/extension.
        
        Right legs (FR, RR): compress during right roll, extend during left roll
        Left legs (FL, RL): extend during right roll, compress during left roll
        
        During neutral phases, legs transition to nominal length.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a right or left leg
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Compute leg length modulation based on phase
        if phase < 0.25:
            # Right roll sway: right legs compress, left legs extend
            phase_local = phase / 0.25
            compression_factor = np.sin(np.pi * phase_local * 0.5)  # 0 -> 1
            if is_right_leg:
                length_delta = -self.leg_compression_ratio * compression_factor
            else:
                length_delta = self.leg_compression_ratio * compression_factor
                
        elif phase < 0.5:
            # Neutral surge 1: transition to nominal
            phase_local = (phase - 0.25) / 0.25
            compression_factor = np.cos(np.pi * phase_local * 0.5)  # 1 -> 0
            if is_right_leg:
                length_delta = -self.leg_compression_ratio * compression_factor
            else:
                length_delta = self.leg_compression_ratio * compression_factor
            # Backward slide in body frame due to forward base velocity
            foot[0] -= 0.04 * np.sin(np.pi * phase_local)
                
        elif phase < 0.75:
            # Left roll sway: left legs compress, right legs extend
            phase_local = (phase - 0.5) / 0.25
            compression_factor = np.sin(np.pi * phase_local * 0.5)  # 0 -> 1
            if is_right_leg:
                length_delta = self.leg_compression_ratio * compression_factor
            else:
                length_delta = -self.leg_compression_ratio * compression_factor
                
        else:
            # Neutral surge 2: transition to nominal
            phase_local = (phase - 0.75) / 0.25
            compression_factor = np.cos(np.pi * phase_local * 0.5)  # 1 -> 0
            if is_right_leg:
                length_delta = self.leg_compression_ratio * compression_factor
            else:
                length_delta = -self.leg_compression_ratio * compression_factor
            # Backward slide in body frame due to forward base velocity
            foot[0] -= 0.04 * np.sin(np.pi * phase_local)
        
        # Apply vertical (z) compression/extension
        foot[2] += foot[2] * length_delta
        
        # Apply slight lateral adjustment to maintain contact during roll
        lateral_adjustment = 0.02 * length_delta
        if is_right_leg:
            foot[1] += lateral_adjustment
        else:
            foot[1] -= lateral_adjustment
        
        return foot