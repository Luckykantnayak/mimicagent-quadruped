from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_BOUND_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Bound gait with alternating front/rear leg pairs.
    
    Phase structure:
      [0.0, 0.40]: Front legs stance, rear legs swing
      [0.40, 0.55]: Transition (double stance, pitch reversal begins earlier)
      [0.55, 0.90]: Rear legs stance, front legs swing
      [0.90, 1.0]: Transition (double stance)
    
    Base motion:
      - Continuous forward velocity
      - Reduced pitch oscillation with earlier reversal to prevent ground penetration
      - Coordinated vertical compensation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Adjusted gait timing parameters (earlier front stance end to begin pitch reversal sooner)
        self.front_stance_end = 0.40
        self.transition_1_end = 0.55
        self.rear_stance_end = 0.90
        self.transition_2_end = 1.0
        
        # Stride parameters
        self.step_length = 0.12
        self.step_height = 0.06
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Reduced base velocity parameters to limit pitch accumulation
        self.vx_base = 0.24
        self.pitch_amplitude = 0.08  # Reduced from 0.15 to limit integrated pitch
        
        # Velocity storage
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Track integrated pitch for compensation
        self.integrated_pitch = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base using steady forward velocity and phase-dependent pitch rate.
        Added coordinated vertical velocity and earlier pitch reversal.
        """
        # Steady forward velocity
        vx = self.vx_base
        
        # Phase-dependent pitch rate with earlier reversal
        if phase < self.front_stance_end:
            # Front stance: nose-down tendency (reduced amplitude)
            pitch_rate = self.pitch_amplitude
            vz = -0.02  # Slight downward velocity during front stance
        elif phase < self.transition_1_end:
            # Transition: smooth reversal from positive to negative
            transition_phase = (phase - self.front_stance_end) / (self.transition_1_end - self.front_stance_end)
            pitch_rate = self.pitch_amplitude * (1.0 - 2.0 * transition_phase)
            vz = -0.02 * (1.0 - transition_phase) + 0.02 * transition_phase
        elif phase < self.rear_stance_end:
            # Rear stance: nose-up tendency
            pitch_rate = -self.pitch_amplitude
            vz = 0.02  # Slight upward velocity during rear stance
        else:
            # Transition: smooth reversal from negative to positive
            transition_phase = (phase - self.rear_stance_end) / (self.transition_2_end - self.rear_stance_end)
            pitch_rate = -self.pitch_amplitude * (1.0 - 2.0 * transition_phase)
            vz = 0.02 * (1.0 - transition_phase) + (-0.02) * transition_phase
        
        # Track integrated pitch for foot compensation
        self.integrated_pitch += pitch_rate * dt
        
        # Set velocity commands
        self.vel_world = np.array([vx, 0.0, vz])
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
        Compute foot trajectory in BODY frame with pitch compensation during stance.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        
        if is_front:
            # Front legs: stance phase [0.0, 0.55], swing phase [0.55, 1.0]
            if phase < self.transition_1_end:
                # Stance phase: foot sweeps backward in body frame
                stance_progress = phase / self.transition_1_end
                foot[0] += self.step_length * (0.5 - stance_progress)
                
                # PITCH COMPENSATION: Lift front feet in body frame to compensate for nose-down pitch
                # As body pitches nose-down, front feet need to be raised in body frame
                # to maintain ground contact in world frame
                pitch_compensation = abs(foot[0]) * np.sin(abs(self.integrated_pitch))
                foot[2] += pitch_compensation
                
            else:
                # Swing phase: foot lifts, swings forward, descends
                swing_progress = (phase - self.transition_1_end) / (self.transition_2_end - self.transition_1_end)
                
                # Longitudinal swing: rear to front
                foot[0] += self.step_length * (swing_progress - 0.5)
                
                # Vertical swing: smooth arc with sine profile
                swing_angle = np.pi * swing_progress
                foot[2] += self.step_height * np.sin(swing_angle)
        else:
            # Rear legs: swing phase [0.0, 0.55], stance phase [0.55, 1.0]
            if phase < self.transition_1_end:
                # Swing phase: foot lifts, swings forward, descends
                swing_progress = phase / self.transition_1_end
                
                # Longitudinal swing: rear to front
                foot[0] += self.step_length * (swing_progress - 0.5)
                
                # Vertical swing: smooth arc with sine profile
                swing_angle = np.pi * swing_progress
                foot[2] += self.step_height * np.sin(swing_angle)
            else:
                # Stance phase: foot sweeps backward in body frame
                stance_progress = (phase - self.transition_1_end) / (self.transition_2_end - self.transition_1_end)
                foot[0] += self.step_length * (0.5 - stance_progress)
                
                # Rear legs during stance benefit from nose-up pitch, so minimal compensation needed
                # Apply slight adjustment to keep symmetric behavior
                pitch_compensation = abs(foot[0]) * np.sin(abs(self.integrated_pitch)) * 0.3
                foot[2] += pitch_compensation
        
        return foot