from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_AERIAL_CROSSOVER_RECOVERY_MotionGenerator(BaseMotionGenerator):
    """
    Aerial Crossover Recovery Skill
    
    Robot performs forward skating glide with alternating diagonal leg pairs
    executing aerial crossover motions. Diagonal pair 1 (FL+RR) and diagonal
    pair 2 (FR+RL) alternate between stance (skating support) and swing
    (aerial crossover wave) phases.
    
    Phase structure:
    - [0.0, 0.4]: FL+RR support, FR+RL aerial crossover
    - [0.4, 0.5]: Transition, all legs in contact
    - [0.5, 0.9]: FR+RL support, FL+RR aerial crossover
    - [0.9, 1.0]: Transition, all legs in contact
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Skating parameters
        self.skating_sweep_length = 0.15
        self.crossover_amplitude_lateral = 0.12
        self.crossover_height = 0.12
        self.crossover_amplitude_forward = 0.08
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity parameters
        self.vx_base = 0.8
        self.vx_enhanced = 1.0
        self.vy_amplitude = 0.05
        self.yaw_rate_amplitude = 0.4

    def update_base_motion(self, phase, dt):
        """
        Update base motion with forward velocity and phase-dependent yaw rate.
        
        Phase [0.0, 0.4]: Positive yaw rate (FR+RL crossover creates CCW moment)
        Phase [0.4, 0.5]: Zero yaw rate (stabilization)
        Phase [0.5, 0.9]: Negative yaw rate (FL+RR crossover creates CW moment)
        Phase [0.9, 1.0]: Zero yaw rate (stabilization)
        """
        
        # Forward velocity increases slightly during second half
        if phase < 0.5:
            vx = self.vx_base
        else:
            vx = self.vx_enhanced
        
        # Lateral velocity oscillates gently
        if phase < 0.4:
            vy = self.vy_amplitude * np.sin(np.pi * phase / 0.4)
        elif phase < 0.5:
            vy = 0.0
        elif phase < 0.9:
            vy = -self.vy_amplitude * np.sin(np.pi * (phase - 0.5) / 0.4)
        else:
            vy = 0.0
        
        # Yaw rate pulses during aerial crossover phases
        if 0.0 <= phase < 0.2:
            yaw_rate = self.yaw_rate_amplitude * np.sin(np.pi * phase / 0.2)
        elif 0.2 <= phase < 0.4:
            yaw_rate = self.yaw_rate_amplitude * np.sin(np.pi * (0.4 - phase) / 0.2)
        elif 0.4 <= phase < 0.5:
            yaw_rate = 0.0
        elif 0.5 <= phase < 0.7:
            yaw_rate = -self.yaw_rate_amplitude * np.sin(np.pi * (phase - 0.5) / 0.2)
        elif 0.7 <= phase < 0.9:
            yaw_rate = -self.yaw_rate_amplitude * np.sin(np.pi * (0.9 - phase) / 0.2)
        else:
            yaw_rate = 0.0
        
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for each leg based on phase.
        
        FL and RR: Stance [0.0, 0.4], transition [0.4, 0.5], swing [0.5, 0.9], transition [0.9, 1.0]
        FR and RL: Swing [0.0, 0.4], transition [0.4, 0.5], stance [0.5, 0.9], transition [0.9, 1.0]
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg group
        is_group1 = leg_name.startswith('FL') or leg_name.startswith('RR')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Define lateral crossover direction
        lateral_sign = 1.0 if is_left else -1.0
        
        if is_group1:
            # FL and RR trajectory
            if phase < 0.4:
                # Stance phase: rearward skating sweep
                progress = phase / 0.4
                foot[0] -= self.skating_sweep_length * (progress - 0.5)
                
            elif phase < 0.5:
                # Transition to swing: reposition forward, maintain contact
                progress = (phase - 0.4) / 0.1
                sweep_pos = self.skating_sweep_length * 0.5
                foot[0] = self.base_feet_pos_body[leg_name][0] - sweep_pos + progress * sweep_pos
                
            elif phase < 0.7:
                # Swing: aerial crossover, inward then upward
                progress = (phase - 0.5) / 0.2
                # Lift off and move inward (toward centerline)
                foot[1] -= lateral_sign * self.crossover_amplitude_lateral * np.sin(np.pi * progress)
                foot[0] += self.crossover_amplitude_forward * progress
                foot[2] += self.crossover_height * np.sin(np.pi * progress)
                
            elif phase < 0.9:
                # Swing: complete crossover arc, move outward and down
                progress = (phase - 0.7) / 0.2
                # Move outward (back toward nominal) and prepare for landing
                foot[1] -= lateral_sign * self.crossover_amplitude_lateral * np.sin(np.pi * (1.0 - progress))
                foot[0] += self.crossover_amplitude_forward * (1.0 + progress)
                foot[2] += self.crossover_height * np.sin(np.pi * (1.0 - progress))
                
            else:
                # Transition to stance: land and begin rearward sweep
                progress = (phase - 0.9) / 0.1
                foot[0] += self.crossover_amplitude_forward * 2.0 * (1.0 - progress)
                
        else:
            # FR and RL trajectory
            if phase < 0.2:
                # Swing: aerial crossover, inward then upward
                progress = phase / 0.2
                foot[1] -= lateral_sign * self.crossover_amplitude_lateral * np.sin(np.pi * progress)
                foot[0] += self.crossover_amplitude_forward * progress
                foot[2] += self.crossover_height * np.sin(np.pi * progress)
                
            elif phase < 0.4:
                # Swing: complete crossover arc, move outward and down
                progress = (phase - 0.2) / 0.2
                foot[1] -= lateral_sign * self.crossover_amplitude_lateral * np.sin(np.pi * (1.0 - progress))
                foot[0] += self.crossover_amplitude_forward * (1.0 + progress)
                foot[2] += self.crossover_height * np.sin(np.pi * (1.0 - progress))
                
            elif phase < 0.5:
                # Transition to stance: land and begin rearward sweep
                progress = (phase - 0.4) / 0.1
                foot[0] += self.crossover_amplitude_forward * 2.0 * (1.0 - progress)
                
            elif phase < 0.9:
                # Stance phase: rearward skating sweep
                progress = (phase - 0.5) / 0.4
                foot[0] -= self.skating_sweep_length * (progress - 0.5)
                
            else:
                # Transition to swing: reposition forward, maintain contact
                progress = (phase - 0.9) / 0.1
                sweep_pos = self.skating_sweep_length * 0.5
                foot[0] = self.base_feet_pos_body[leg_name][0] - sweep_pos + progress * sweep_pos
        
        return foot