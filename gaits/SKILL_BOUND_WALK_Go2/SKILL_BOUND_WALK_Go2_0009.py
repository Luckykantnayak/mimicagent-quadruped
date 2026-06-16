from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_BOUND_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Bound gait with alternating front/rear leg pairs.
    
    Phase structure:
      [0.0, 0.40]: Front legs stance, rear legs swing
      [0.40, 0.55]: Transition (double stance, pitch reversal)
      [0.55, 0.90]: Rear legs stance, front legs swing
      [0.90, 1.0]: Transition (double stance)
    
    Base motion:
      - Continuous forward velocity
      - Reduced pitch oscillation with earlier reversal
      - Pitch compensation extracted directly from quaternion to prevent drift
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Adjusted gait timing parameters
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
        
        # Base velocity parameters
        self.vx_base = 0.24
        self.pitch_amplitude = 0.08
        
        # Velocity storage
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def extract_pitch_from_quaternion(self, quat):
        """
        Extract pitch (rotation around Y axis) from quaternion [w, x, y, z].
        Returns pitch in radians.
        """
        w, x, y, z = quat
        # Pitch = arcsin(2*(w*y - z*x))
        sin_pitch = 2.0 * (w * y - z * x)
        # Clamp to avoid numerical issues with arcsin
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
        pitch = np.arcsin(sin_pitch)
        return pitch

    def update_base_motion(self, phase, dt):
        """
        Update base using steady forward velocity and phase-dependent pitch rate.
        Pitch is extracted from quaternion for accurate compensation downstream.
        """
        # Steady forward velocity
        vx = self.vx_base
        
        # Phase-dependent pitch rate with earlier reversal
        if phase < self.front_stance_end:
            # Front stance: nose-down tendency
            pitch_rate = self.pitch_amplitude
            vz = -0.02
        elif phase < self.transition_1_end:
            # Transition: smooth reversal from positive to negative
            transition_phase = (phase - self.front_stance_end) / (self.transition_1_end - self.front_stance_end)
            pitch_rate = self.pitch_amplitude * (1.0 - 2.0 * transition_phase)
            vz = -0.02 * (1.0 - transition_phase) + 0.02 * transition_phase
        elif phase < self.rear_stance_end:
            # Rear stance: nose-up tendency
            pitch_rate = -self.pitch_amplitude
            vz = 0.02
        else:
            # Transition: smooth reversal from negative to positive
            transition_phase = (phase - self.rear_stance_end) / (self.transition_2_end - self.rear_stance_end)
            pitch_rate = -self.pitch_amplitude * (1.0 - 2.0 * transition_phase)
            vz = 0.02 * (1.0 - transition_phase) + (-0.02) * transition_phase
        
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
        Extracts current pitch directly from quaternion to avoid drift.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Extract actual current pitch from body orientation
        current_pitch = self.extract_pitch_from_quaternion(self.root_quat)
        
        is_front = leg_name.startswith('F')
        
        if is_front:
            # Front legs: stance phase [0.0, 0.55], swing phase [0.55, 1.0]
            if phase < self.transition_1_end:
                # Stance phase: foot sweeps backward in body frame
                stance_progress = phase / self.transition_1_end
                foot[0] += self.step_length * (0.5 - stance_progress)
                
                # PITCH COMPENSATION: Lift front feet in body frame to compensate for nose-down pitch
                # When pitch is positive (nose-down), front feet (positive x) need upward compensation
                # Compensation magnitude: x_position * sin(pitch)
                if current_pitch > 0:
                    pitch_compensation = foot[0] * np.sin(current_pitch)
                    foot[2] += pitch_compensation
                
            else:
                # Swing phase: foot lifts, swings forward, descends
                swing_progress = (phase - self.transition_1_end) / (self.transition_2_end - self.transition_1_end)
                
                # Longitudinal swing: rear to front
                foot[0] += self.step_length * (swing_progress - 0.5)
                
                # Vertical swing: smooth arc
                swing_angle = np.pi * swing_progress
                foot[2] += self.step_height * np.sin(swing_angle)
        else:
            # Rear legs: swing phase [0.0, 0.55], stance phase [0.55, 1.0]
            if phase < self.transition_1_end:
                # Swing phase: foot lifts, swings forward, descends
                swing_progress = phase / self.transition_1_end
                
                # Longitudinal swing: rear to front
                foot[0] += self.step_length * (swing_progress - 0.5)
                
                # Vertical swing: smooth arc
                swing_angle = np.pi * swing_progress
                foot[2] += self.step_height * np.sin(swing_angle)
            else:
                # Stance phase: foot sweeps backward in body frame
                stance_progress = (phase - self.transition_1_end) / (self.transition_2_end - self.transition_1_end)
                foot[0] += self.step_length * (0.5 - stance_progress)
                
                # Rear legs during rear stance: when pitch is negative (nose-up), rear feet may need slight adjustment
                # Rear feet have negative x in body frame, and nose-up pitch (negative) can lower them
                if current_pitch < 0:
                    # For rear legs, negative pitch means nose-up, which raises the rear
                    # Apply minimal compensation as rear benefits from this
                    pitch_compensation = abs(foot[0]) * np.sin(abs(current_pitch)) * 0.2
                    foot[2] += pitch_compensation
        
        return foot