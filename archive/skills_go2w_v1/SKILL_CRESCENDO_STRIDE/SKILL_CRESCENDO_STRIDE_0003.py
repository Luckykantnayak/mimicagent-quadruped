from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRESCENDO_STRIDE_MotionGenerator(BaseMotionGenerator):
    """
    Crescendo Stride: A trot gait with stride length and velocity smoothly
    crescendoing from minimal to maximal over phases 0→0.8, then abruptly
    resetting during phases 0.8→1.0.
    
    - Diagonal pair trot coordination (FL+RR vs FR+RL)
    - Base forward velocity modulated as function of phase
    - Foot swing amplitude scales with velocity to maintain coordination
    - No lateral or yaw motion
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Diagonal trot phase offsets
        # Group 1 (FL, RR): phase offset 0.0
        # Group 2 (FR, RL): phase offset 0.5
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0
            else:  # FR or RL
                self.phase_offsets[leg] = 0.5
        
        # Stride parameters
        self.duty_cycle = 0.5  # 50% stance, 50% swing for trot
        
        # Velocity crescendo parameters
        self.vx_min = 0.2   # Minimal forward velocity (m/s)
        self.vx_max = 1.2   # Maximum forward velocity (m/s)
        
        # Stride length parameters (emerge from velocity integration)
        # These control foot swing amplitude in body frame
        self.step_length_min = 0.05  # Minimal stride extension
        self.step_length_max = 0.25  # Maximum stride extension
        
        # Step height parameters (scale with stride)
        self.step_height_min = 0.04
        self.step_height_max = 0.12
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Current velocity (updated each step)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def compute_velocity_profile(self, phase):
        """
        Compute forward velocity as function of phase.
        
        Phase ranges:
        - [0.0, 0.2]: minimal (20% of max)
        - [0.2, 0.4]: ramp 20% → 50%
        - [0.4, 0.6]: ramp 50% → 85%
        - [0.6, 0.8]: sustain 100%
        - [0.8, 1.0]: rapid decay 100% → 20%
        """
        if phase < 0.2:
            # Minimal stride phase
            velocity_factor = 0.2
        elif phase < 0.4:
            # Building stride: 20% → 50%
            local_phase = (phase - 0.2) / 0.2
            velocity_factor = 0.2 + 0.3 * local_phase
        elif phase < 0.6:
            # Accelerating stride: 50% → 85%
            local_phase = (phase - 0.4) / 0.2
            velocity_factor = 0.5 + 0.35 * local_phase
        elif phase < 0.8:
            # Peak stride: 85% → 100%
            local_phase = (phase - 0.6) / 0.2
            velocity_factor = 0.85 + 0.15 * local_phase
        else:
            # Reset stride: 100% → 20% (rapid deceleration)
            local_phase = (phase - 0.8) / 0.2
            velocity_factor = 1.0 - 0.8 * local_phase
        
        vx = self.vx_min + (self.vx_max - self.vx_min) * velocity_factor
        return vx

    def compute_stride_parameters(self, phase):
        """
        Compute stride length and height based on velocity profile.
        These scale foot swing amplitude to match base motion.
        """
        velocity_factor = (self.compute_velocity_profile(phase) - self.vx_min) / (self.vx_max - self.vx_min)
        
        step_length = self.step_length_min + (self.step_length_max - self.step_length_min) * velocity_factor
        step_height = self.step_height_min + (self.step_height_max - self.step_height_min) * velocity_factor
        
        return step_length, step_height

    def update_base_motion(self, phase, dt):
        """
        Update base velocity command based on crescendo profile.
        Position integrates from velocity (handled by base class).
        """
        vx = self.compute_velocity_profile(phase)
        
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot trajectory in body frame for given leg and phase.
        
        Trot pattern:
        - Each leg has 50% stance, 50% swing
        - Diagonal pairs synchronized (phase offset 0.0 or 0.5)
        - Swing amplitude scales with velocity crescendo
        """
        # Get leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Get current stride parameters
        step_length, step_height = self.compute_stride_parameters(phase)
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_phase < self.duty_cycle:
            # Stance phase: foot sweeps rearward relative to body
            # Progress from 0 (front) to 1 (rear)
            progress = leg_phase / self.duty_cycle
            foot[0] += step_length * (0.5 - progress)
        else:
            # Swing phase: foot lifts and swings forward
            # Progress from 0 (rear, liftoff) to 1 (front, touchdown)
            progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)
            
            # Forward swing: rear → front
            foot[0] += step_length * (progress - 0.5)
            
            # Vertical arc: sine wave for smooth liftoff/touchdown
            swing_angle = np.pi * progress
            foot[2] += step_height * np.sin(swing_angle)
        
        return foot