from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_STUTTER_STEP_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Stutter-step forward locomotion with three discrete forward pulses
    separated by brief stabilization pauses.
    
    Phase structure:
      [0.00, 0.15]: pulse_1_all_legs - all legs push simultaneously
      [0.15, 0.30]: pause_1_lock - all legs locked, base decelerates
      [0.30, 0.45]: pulse_2_diagonal_FL_RR - FL-RR drive, FR-RL stabilize
      [0.45, 0.60]: pause_2_pitch_oscillation - all legs locked with pitch oscillation
      [0.60, 0.80]: pulse_3_diagonal_FR_RL - FR-RL drive, FL-RR stabilize
      [0.80, 1.00]: pause_3_stabilization - final rest, prepare for cycle restart
    
    All legs maintain ground contact throughout. Pulses are executed via
    rapid leg extension (push) while maintaining contact.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Leg extension parameters (body frame displacement during push)
        self.push_distance = 0.12  # Maximum backward displacement during push
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters
        self.vx_pulse_peak = 0.8  # Peak forward velocity during pulses (m/s)
        self.pitch_osc_amp = 0.15  # Pitch oscillation amplitude during pause_2 (rad)
        self.pitch_osc_freq = 8.0  # Pitch oscillation frequency (Hz)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on current phase.
        Three forward pulses with pauses, pitch oscillation during pause_2.
        """
        vx = 0.0
        pitch_rate = 0.0
        
        # Pulse 1: all legs push [0.0, 0.15]
        if 0.0 <= phase < 0.15:
            # Rapid acceleration to peak
            progress = phase / 0.15
            vx = self.vx_pulse_peak * np.sin(np.pi * progress)
            # Slight pitch-down during thrust
            pitch_rate = -0.5 * np.sin(np.pi * progress)
        
        # Pause 1: deceleration and lock [0.15, 0.3]
        elif 0.15 <= phase < 0.3:
            # Rapid deceleration
            progress = (phase - 0.15) / 0.15
            vx = self.vx_pulse_peak * 0.3 * (1.0 - progress)
            # Pitch-up as base settles
            pitch_rate = 0.4 * progress
        
        # Pulse 2: FL-RR diagonal push [0.3, 0.45]
        elif 0.3 <= phase < 0.45:
            progress = (phase - 0.3) / 0.15
            vx = self.vx_pulse_peak * 0.85 * np.sin(np.pi * progress)
            # Slight pitch-down during drive
            pitch_rate = -0.4 * np.sin(np.pi * progress)
        
        # Pause 2: pitch oscillation [0.45, 0.6]
        elif 0.45 <= phase < 0.6:
            progress = (phase - 0.45) / 0.15
            vx = self.vx_pulse_peak * 0.25 * (1.0 - progress)
            # Pitch oscillation for dynamic balancing
            osc_phase = 2.0 * np.pi * progress
            pitch_rate = self.pitch_osc_amp * self.pitch_osc_freq * np.cos(osc_phase)
        
        # Pulse 3: FR-RL diagonal push [0.6, 0.8]
        elif 0.6 <= phase < 0.8:
            progress = (phase - 0.6) / 0.2
            vx = self.vx_pulse_peak * 0.75 * np.sin(np.pi * progress)
            # Slight pitch-down during drive
            pitch_rate = -0.35 * np.sin(np.pi * progress)
        
        # Pause 3: final stabilization [0.8, 1.0]
        elif 0.8 <= phase <= 1.0:
            progress = (phase - 0.8) / 0.2
            vx = self.vx_pulse_peak * 0.2 * (1.0 - progress)
            # Converge pitch rate to zero
            pitch_rate = -0.2 * (1.0 - progress)
        
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
        Compute foot position in body frame based on phase and leg coordination.
        
        All legs maintain contact. During push phases, active legs extend backward
        (in body frame) to generate forward thrust. During pause phases, legs hold
        position or reposition smoothly.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_FL = leg_name.startswith('FL')
        is_FR = leg_name.startswith('FR')
        is_RL = leg_name.startswith('RL')
        is_RR = leg_name.startswith('RR')
        
        # Pulse 1: all legs push [0.0, 0.15]
        if 0.0 <= phase < 0.15:
            progress = phase / 0.15
            # All legs extend backward in body frame
            foot[0] -= self.push_distance * progress
        
        # Pause 1: hold position [0.15, 0.3]
        elif 0.15 <= phase < 0.3:
            # Maintain extended position
            foot[0] -= self.push_distance
        
        # Pulse 2: FL-RR drive, FR-RL stabilize [0.3, 0.45]
        elif 0.3 <= phase < 0.45:
            progress = (phase - 0.3) / 0.15
            if is_FL or is_RR:
                # FL-RR diagonal: continue extending backward
                foot[0] -= self.push_distance * (1.0 + 0.7 * progress)
            else:
                # FR-RL diagonal: start repositioning forward
                foot[0] -= self.push_distance * (1.0 - 0.3 * progress)
        
        # Pause 2: hold with pitch oscillation compliance [0.45, 0.6]
        elif 0.45 <= phase < 0.6:
            if is_FL or is_RR:
                # FL-RR hold extended position
                foot[0] -= self.push_distance * 1.7
                # Small vertical adjustment for pitch compliance
                osc_phase = 2.0 * np.pi * (phase - 0.45) / 0.15
                foot[2] += 0.005 * np.sin(osc_phase)
            else:
                # FR-RL continue gradual forward repositioning
                progress = (phase - 0.45) / 0.15
                foot[0] -= self.push_distance * (0.7 - 0.4 * progress)
                foot[2] += 0.005 * np.sin(osc_phase)
        
        # Pulse 3: FR-RL drive, FL-RR stabilize [0.6, 0.8]
        elif 0.6 <= phase < 0.8:
            progress = (phase - 0.6) / 0.2
            if is_FR or is_RL:
                # FR-RL diagonal: extend backward
                foot[0] -= self.push_distance * (0.3 + 0.9 * progress)
            else:
                # FL-RR diagonal: reposition forward
                foot[0] -= self.push_distance * (1.7 - 0.9 * progress)
        
        # Pause 3: return to neutral [0.8, 1.0]
        elif 0.8 <= phase <= 1.0:
            progress = (phase - 0.8) / 0.2
            if is_FR or is_RL:
                # FR-RL return to neutral
                foot[0] -= self.push_distance * (1.2 - 1.2 * progress)
            else:
                # FL-RR return to neutral
                foot[0] -= self.push_distance * (0.8 - 0.8 * progress)
        
        return foot