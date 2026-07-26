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
        
        self.duty_cycle = 0.5
        
        self.max_step_height = 0.12
        self.max_step_length = 0.15
        
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0
            else:
                self.phase_offsets[leg] = 0.5
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.max_vx = 1.2
        self.min_vx = 0.15
        
        # Reduced pitch rate to minimize ground penetration
        self.max_pitch_rate = np.deg2rad(3.0)
        
        # Track touchdown world Z for each leg to lock stance phase
        self.touchdown_z_world = {leg: 0.0 for leg in leg_names}
        self.prev_leg_phase = {leg: 0.0 for leg in leg_names}

    def smooth_sigmoid(self, x):
        """Smooth sigmoid-like transition function."""
        return 3.0 * x**2 - 2.0 * x**3

    def compute_velocity_scale(self, phase):
        """
        Compute velocity scaling factor based on phase with smooth transitions.
        
        0.0-0.2: 10-20% (minimal)
        0.2-0.4: ramp to 50%
        0.4-0.6: ramp to 75%
        0.6-0.8: ramp to 100% (peak)
        0.8-1.0: smooth drop back to 10-20%
        """
        if phase < 0.2:
            progress = phase / 0.2
            smooth_progress = self.smooth_sigmoid(progress)
            return 0.1 + 0.1 * smooth_progress
        elif phase < 0.4:
            progress = (phase - 0.2) / 0.2
            smooth_progress = self.smooth_sigmoid(progress)
            return 0.2 + 0.3 * smooth_progress
        elif phase < 0.6:
            progress = (phase - 0.4) / 0.2
            smooth_progress = self.smooth_sigmoid(progress)
            return 0.5 + 0.25 * smooth_progress
        elif phase < 0.8:
            progress = (phase - 0.6) / 0.2
            smooth_progress = self.smooth_sigmoid(progress)
            return 0.75 + 0.25 * smooth_progress
        else:
            # Smooth polynomial decrescendo instead of exponential
            progress = (phase - 0.8) / 0.2
            smooth_progress = self.smooth_sigmoid(progress)
            return 1.0 - 0.85 * smooth_progress

    def compute_stride_amplitude_scale(self, phase):
        """
        Compute stride amplitude scaling (step height and reach) based on phase.
        Follows same profile as velocity for coordinated crescendo effect.
        """
        return self.compute_velocity_scale(phase)

    def compute_pitch_rate(self, phase):
        """
        Compute pitch rate based on phase to reflect acceleration/deceleration.
        Reduced magnitude to prevent ground penetration.
        """
        if phase < 0.2:
            return 0.0
        elif phase < 0.8:
            progress = (phase - 0.2) / 0.6
            smooth_progress = self.smooth_sigmoid(progress)
            return self.max_pitch_rate * np.sin(np.pi * smooth_progress * 0.5)
        else:
            progress = (phase - 0.8) / 0.2
            smooth_progress = self.smooth_sigmoid(progress)
            return -self.max_pitch_rate * np.sin(np.pi * smooth_progress)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on crescendo profile.
        """
        vel_scale = self.compute_velocity_scale(phase)
        vx = self.min_vx + (self.max_vx - self.min_vx) * vel_scale
        
        pitch_rate = self.compute_pitch_rate(phase)
        
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
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
        Stance phase Z is locked in world frame to prevent ground penetration.
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        amplitude_scale = self.compute_stride_amplitude_scale(phase)
        vel_scale = self.compute_velocity_scale(phase)
        
        step_height = self.max_step_height * amplitude_scale
        step_length = self.max_step_length * amplitude_scale
        
        # Detect stance-swing transition to record touchdown Z
        prev_phase = self.prev_leg_phase[leg_name]
        if prev_phase >= self.duty_cycle and leg_phase < self.duty_cycle:
            # Just transitioned from swing to stance - record touchdown
            foot_world = quat_rotate(self.root_quat, foot) + self.root_pos
            self.touchdown_z_world[leg_name] = foot_world[2]
        self.prev_leg_phase[leg_name] = leg_phase
        
        if leg_phase < self.duty_cycle:
            # Stance phase: foot moves backward, Z locked in world frame
            progress = leg_phase / self.duty_cycle
            smooth_progress = self.smooth_sigmoid(progress)
            
            # Backward motion scaled by velocity to match base forward motion
            # Retraction distance accounts for instantaneous velocity
            foot[0] += step_length * (0.5 - smooth_progress) * vel_scale / amplitude_scale if amplitude_scale > 0.01 else 0.0
            
            # Lock Z in world frame by compensating for base orientation
            foot_world = quat_rotate(self.root_quat, foot) + self.root_pos
            z_error = foot_world[2] - self.touchdown_z_world[leg_name]
            
            # Transform Z correction back to body frame
            quat_inv = np.array([self.root_quat[0], -self.root_quat[1], -self.root_quat[2], -self.root_quat[3]])
            z_correction_world = np.array([0.0, 0.0, -z_error])
            z_correction_body = quat_rotate(quat_inv, z_correction_world)
            foot += z_correction_body
            
        else:
            # Swing phase: foot lifts, swings forward, and lands
            swing_progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)
            smooth_swing = self.smooth_sigmoid(swing_progress)
            
            foot[0] += step_length * (smooth_swing - 0.5)
            
            # Smooth parabolic arc with continuous derivatives
            arc_progress = np.sin(np.pi * swing_progress)
            foot[2] += step_height * arc_progress
        
        return foot