from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_BOUND_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Bound gait with forward locomotion and pitch oscillation.
    
    - Front legs (FL, FR) stance in [0.0, 0.5], swing in [0.5, 1.0]
    - Rear legs (RL, RR) swing in [0.0, 0.5], stance in [0.5, 1.0]
    - Base moves forward continuously with mild pitch oscillation
    - Pitch tilts nose-down during front stance, nose-up during rear stance
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Gait parameters
        self.step_length = 0.12  # Forward reach during swing
        self.step_height = 0.10  # Ground clearance during swing
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets: front legs at 0.0, rear legs at 0.5 (anti-phase)
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('F'):  # Front legs
                self.phase_offsets[leg] = 0.0
            elif leg.startswith('R'):  # Rear legs
                self.phase_offsets[leg] = 0.5
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity parameters
        self.vx_forward = 0.6  # Steady forward velocity
        self.vz_amplitude = 0.05  # Small vertical velocity pulse for clearance
        self.pitch_rate_amplitude = 0.2  # Pitch oscillation magnitude

    def update_base_motion(self, phase, dt):
        """
        Update base with steady forward velocity and pitch oscillation.
        
        Phase [0.0, 0.5]: front stance, pitch nose-down (negative pitch rate)
        Phase [0.5, 1.0]: rear stance, pitch nose-up (positive pitch rate)
        
        Small vertical velocity pulses near transitions to assist swing clearance.
        """
        
        # Steady forward velocity
        vx = self.vx_forward
        
        # Small vertical velocity pulse near phase transitions for clearance
        # Pulse near 0.5 (rear legs about to land) and near 0.0/1.0 (front legs about to land)
        vz = 0.0
        if 0.4 < phase < 0.6:
            # Transition from front to rear stance
            vz = self.vz_amplitude * np.sin(np.pi * (phase - 0.4) / 0.2)
        elif phase > 0.9 or phase < 0.1:
            # Transition from rear to front stance
            phase_shifted = (phase + 0.1) % 1.0
            if phase_shifted < 0.2:
                vz = self.vz_amplitude * np.sin(np.pi * phase_shifted / 0.2)
        
        # Pitch rate: negative (nose down) in first half, positive (nose up) in second half
        # Smooth sinusoidal variation
        pitch_rate = self.pitch_rate_amplitude * np.sin(2 * np.pi * phase + np.pi)
        # This gives: phase=0 -> 0, phase=0.25 -> -max, phase=0.5 -> 0, phase=0.75 -> +max, phase=1.0 -> 0
        
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
        Compute foot position in body frame based on stance/swing phase.
        
        Front legs (FL, FR):
          - Stance [0.0, 0.5]: foot slides rearward as base moves forward
          - Swing [0.5, 1.0]: foot arcs forward and upward
        
        Rear legs (RL, RR):
          - Swing [0.0, 0.5]: foot arcs forward and upward
          - Stance [0.5, 1.0]: foot slides rearward as base moves forward
        """
        
        # Get leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this leg is in stance or swing
        # Stance is [0.0, 0.5] for each leg's local phase
        in_stance = leg_phase < 0.5
        
        if in_stance:
            # Stance phase: foot slides rearward in body frame
            # At leg_phase=0.0, foot is at forward position
            # At leg_phase=0.5, foot is at rearward position
            stance_progress = leg_phase / 0.5
            foot[0] += self.step_length * (0.5 - stance_progress)
            # Keep foot on ground
            foot[2] = self.base_feet_pos_body[leg_name][2]
            
        else:
            # Swing phase: foot lifts, swings forward, descends
            # leg_phase in [0.5, 1.0]
            swing_progress = (leg_phase - 0.5) / 0.5
            
            # Horizontal: swing from rearward to forward position
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # Vertical: arc trajectory using sine for smooth lift and landing
            foot[2] += self.step_height * np.sin(np.pi * swing_progress)
        
        return foot