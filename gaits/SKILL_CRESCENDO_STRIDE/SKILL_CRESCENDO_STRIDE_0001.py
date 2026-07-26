from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRESCENDO_STRIDE_MotionGenerator(BaseMotionGenerator):
    """
    Crescendo Stride: A diagonal trot gait with smoothly ramping stride amplitude and velocity.
    
    Phase 0.0-0.2: Minimal stride (10-20% velocity, low step height)
    Phase 0.2-0.4: Early crescendo (ramping to 50% velocity)
    Phase 0.4-0.6: Mid crescendo (75% velocity, high clearance)
    Phase 0.6-0.8: Peak stride (100% velocity, maximum step height and reach)
    Phase 0.8-1.0: Sudden decrescendo (collapse back to minimal)
    
    Diagonal pairs: FL+RR vs FR+RL alternate every ~0.1 phase.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Trot timing: each diagonal pair alternates stance/swing every 0.1 phase
        # Total cycle = 1.0, so each leg completes stance and swing once per cycle
        self.duty_cycle = 0.5  # 50% stance, 50% swing for each leg
        
        # Maximum stride parameters (achieved at peak phase 0.6-0.8)
        self.max_step_height = 0.12
        self.max_step_length = 0.15
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for diagonal trot coordination
        # FL and RR move together (offset 0.0)
        # FR and RL move together (offset 0.5)
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0
            else:  # FR or RL
                self.phase_offsets[leg] = 0.5
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity parameters
        self.max_vx = 1.2  # Maximum forward velocity (m/s)
        self.min_vx = 0.15  # Minimum forward velocity (10-20% of max)
        
        # Pitch rate parameters (degrees/s converted to rad/s)
        self.max_pitch_rate = np.deg2rad(15.0)

    def compute_velocity_scale(self, phase):
        """
        Compute velocity scaling factor based on phase.
        
        0.0-0.2: 10-20% (minimal)
        0.2-0.4: ramp to 50%
        0.4-0.6: ramp to 75%
        0.6-0.8: ramp to 100% (peak)
        0.8-1.0: sharp drop back to 10-20%
        """
        if phase < 0.2:
            # Minimal phase: linear ramp from 10% to 20%
            return 0.1 + 0.1 * (phase / 0.2)
        elif phase < 0.4:
            # Early crescendo: 20% to 50%
            progress = (phase - 0.2) / 0.2
            return 0.2 + 0.3 * progress
        elif phase < 0.6:
            # Mid crescendo: 50% to 75%
            progress = (phase - 0.4) / 0.2
            return 0.5 + 0.25 * progress
        elif phase < 0.8:
            # Peak stride: 75% to 100%
            progress = (phase - 0.6) / 0.2
            return 0.75 + 0.25 * progress
        else:
            # Sudden decrescendo: 100% to 15% (steep exponential drop)
            progress = (phase - 0.8) / 0.2
            # Exponential decay from 1.0 to 0.15
            return 0.15 + 0.85 * np.exp(-5.0 * progress)

    def compute_stride_amplitude_scale(self, phase):
        """
        Compute stride amplitude scaling (step height and reach) based on phase.
        Follows same profile as velocity for coordinated crescendo effect.
        """
        return self.compute_velocity_scale(phase)

    def compute_pitch_rate(self, phase):
        """
        Compute pitch rate based on phase to reflect acceleration/deceleration.
        
        0.0-0.2: near zero (minimal motion)
        0.2-0.8: positive (pitching forward during crescendo)
        0.8-1.0: negative (pitching backward during decrescendo)
        """
        if phase < 0.2:
            return 0.0
        elif phase < 0.8:
            # During crescendo: gentle forward pitch
            # Peak at mid-crescendo
            progress = (phase - 0.2) / 0.6
            return self.max_pitch_rate * np.sin(np.pi * progress * 0.5)
        else:
            # During decrescendo: pitch backward to stabilize
            progress = (phase - 0.8) / 0.2
            return -self.max_pitch_rate * np.sin(np.pi * progress)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on crescendo profile.
        """
        # Compute forward velocity based on phase
        vel_scale = self.compute_velocity_scale(phase)
        vx = self.min_vx + (self.max_vx - self.min_vx) * vel_scale
        
        # Compute pitch rate
        pitch_rate = self.compute_pitch_rate(phase)
        
        # Set velocity commands
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
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
        Compute foot trajectory in body frame for diagonal trot with crescendo.
        
        Each leg alternates between stance (0.5 phase duration) and swing (0.5 phase duration).
        Swing amplitude (height and forward reach) scales with the crescendo profile.
        """
        # Compute leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Get base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute amplitude scaling for this global phase
        amplitude_scale = self.compute_stride_amplitude_scale(phase)
        
        # Scaled stride parameters
        step_height = self.max_step_height * amplitude_scale
        step_length = self.max_step_length * amplitude_scale
        
        # Determine if in stance or swing
        if leg_phase < self.duty_cycle:
            # Stance phase: foot moves backward relative to body as body moves forward
            progress = leg_phase / self.duty_cycle
            # Linear backward motion from front to back of stance
            foot[0] += step_length * (0.5 - progress)
        else:
            # Swing phase: foot lifts, swings forward, and lands
            swing_progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)
            
            # Forward reach: sinusoidal from back to front
            foot[0] += step_length * (swing_progress - 0.5)
            
            # Vertical clearance: parabolic arc
            # Maximum height at mid-swing (swing_progress = 0.5)
            foot[2] += step_height * 4.0 * swing_progress * (1.0 - swing_progress)
        
        return foot