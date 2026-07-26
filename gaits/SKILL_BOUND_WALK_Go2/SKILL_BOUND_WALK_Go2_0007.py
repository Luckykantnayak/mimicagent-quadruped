from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_BOUND_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Bound gait with alternating front/rear leg pairs.
    
    Phase structure:
      [0.0, 0.45]: Front legs stance, rear legs swing
      [0.45, 0.55]: Transition (double stance)
      [0.55, 0.95]: Rear legs stance, front legs swing
      [0.95, 1.0]: Transition (double stance)
    
    Base motion:
      - Continuous forward velocity
      - Pitch oscillation: nose-down during front stance, nose-up during rear stance
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Gait timing parameters
        self.front_stance_end = 0.45
        self.transition_1_end = 0.55
        self.rear_stance_end = 0.95
        self.transition_2_end = 1.0
        
        # Stride parameters
        self.step_length = 0.12  # Forward stride length per leg
        self.step_height = 0.06  # Swing clearance height
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base velocity parameters
        self.vx_base = 0.24  # Steady forward velocity (m/s)
        self.pitch_amplitude = 0.15  # Pitch oscillation amplitude (rad/s peak)
        
        # Velocity storage
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base using steady forward velocity and phase-dependent pitch rate.
        
        Pitch dynamics:
          - [0.0, 0.45]: Front stance → pitch nose-down (positive pitch rate)
          - [0.45, 0.55]: Transition → pitch rate reverses through zero
          - [0.55, 0.95]: Rear stance → pitch nose-up (negative pitch rate)
          - [0.95, 1.0]: Transition → pitch rate reverses through zero
        """
        # Steady forward velocity
        vx = self.vx_base
        
        # Phase-dependent pitch rate
        if phase < self.front_stance_end:
            # Front stance: nose-down tendency
            pitch_rate = self.pitch_amplitude
        elif phase < self.transition_1_end:
            # Transition: smooth reversal from positive to negative
            transition_phase = (phase - self.front_stance_end) / (self.transition_1_end - self.front_stance_end)
            pitch_rate = self.pitch_amplitude * (1.0 - 2.0 * transition_phase)
        elif phase < self.rear_stance_end:
            # Rear stance: nose-up tendency
            pitch_rate = -self.pitch_amplitude
        else:
            # Transition: smooth reversal from negative to positive
            transition_phase = (phase - self.rear_stance_end) / (self.transition_2_end - self.rear_stance_end)
            pitch_rate = -self.pitch_amplitude * (1.0 - 2.0 * transition_phase)
        
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
        Compute foot trajectory in BODY frame based on leg pair and phase.
        
        Front legs (FL, FR):
          - Stance: [0.0, 0.55] (including transition)
          - Swing: [0.55, 1.0]
        
        Rear legs (RL, RR):
          - Swing: [0.0, 0.55]
          - Stance: [0.55, 1.0] (including transition)
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        
        if is_front:
            # Front legs: stance phase [0.0, 0.55], swing phase [0.55, 1.0]
            if phase < self.transition_1_end:
                # Stance phase: foot sweeps backward in body frame
                stance_progress = phase / self.transition_1_end
                foot[0] += self.step_length * (0.5 - stance_progress)
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
        
        return foot