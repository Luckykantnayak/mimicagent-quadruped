from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_SKIP_SYNCOPATION_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Syncopated forward skipping motion with double-bounces, extended glides, and diagonal landing.
    
    Phase structure:
    - [0.0, 0.15]: Rear double-bounce
    - [0.15, 0.35]: Long forward glide (all legs airborne)
    - [0.35, 0.4]: Front brief contact
    - [0.4, 0.55]: Front double-bounce
    - [0.55, 0.8]: Extended asymmetric glide (all legs airborne)
    - [0.8, 0.88]: First diagonal pair (FL+RR) lands
    - [0.88, 1.0]: Second diagonal pair (FR+RL) lands (syncopated)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.2
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.step_length = 0.25
        self.rear_bounce_height = 0.12
        self.front_bounce_height = 0.10
        self.glide_extension_height = 0.15
        self.double_bounce_frequency = 15.0
        self.double_bounce_amplitude = 0.06
        
        # Base velocity parameters
        self.forward_velocity = 0.8
        self.rear_bounce_vz = 0.5
        self.front_bounce_vz = 0.4
        self.rear_bounce_pitch_rate = -1.2
        self.front_bounce_pitch_rate = 1.5
        self.landing_roll_amplitude = 0.8
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion according to phase-specific velocity commands.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        if 0.0 <= phase < 0.15:
            # Rear double-bounce: forward and upward, pitch down
            vx = self.forward_velocity
            vz = self.rear_bounce_vz
            pitch_rate = self.rear_bounce_pitch_rate
            
        elif 0.15 <= phase < 0.35:
            # Long forward glide: ballistic trajectory
            vx = self.forward_velocity
            # Ballistic arc: upward at start, downward at end
            glide_progress = (phase - 0.15) / 0.2
            vz = self.rear_bounce_vz * (1.0 - 2.0 * glide_progress)
            pitch_rate = 0.0
            
        elif 0.35 <= phase < 0.4:
            # Front brief contact: forward with downward, deceleration
            vx = self.forward_velocity * 0.9
            vz = -0.3
            pitch_rate = 0.0
            
        elif 0.4 <= phase < 0.55:
            # Front double-bounce: forward and upward, pitch up
            vx = self.forward_velocity
            vz = self.front_bounce_vz
            pitch_rate = self.front_bounce_pitch_rate
            
        elif 0.55 <= phase < 0.8:
            # Extended asymmetric glide: ballistic with pitch adjustment
            vx = self.forward_velocity
            glide_progress = (phase - 0.55) / 0.25
            vz = self.front_bounce_vz * (1.0 - 2.0 * glide_progress)
            # Pitch gradually levels out
            pitch_rate = self.front_bounce_pitch_rate * (1.0 - glide_progress)
            
        elif 0.8 <= phase < 1.0:
            # Syncopated landing: forward with downward, roll oscillation
            vx = self.forward_velocity
            vz = -0.4
            # Roll oscillation due to diagonal landing
            landing_progress = (phase - 0.8) / 0.2
            roll_rate = self.landing_roll_amplitude * np.sin(2.0 * np.pi * landing_progress)
            pitch_rate = 0.0
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase and leg.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        is_rear = leg_name.startswith('R')
        is_left = leg_name.endswith('L')
        is_fl = leg_name == 'FL'
        is_fr = leg_name == 'FR'
        is_rl = leg_name == 'RL'
        is_rr = leg_name == 'RR'
        
        # Phase 1: Rear double-bounce [0.0, 0.15]
        if 0.0 <= phase < 0.15:
            if is_rear:
                # Stance with double-bounce oscillation
                progress = phase / 0.15
                double_bounce_phase = progress * 2.0 * np.pi * 2.0  # Two bounces
                compression = self.double_bounce_amplitude * (1.0 - np.cos(double_bounce_phase))
                foot[2] -= compression
                foot[0] -= self.step_length * 0.3
            else:  # Front legs
                # Swing forward and upward
                progress = phase / 0.15
                foot[0] += self.step_length * progress
                foot[2] += self.rear_bounce_height * progress
        
        # Phase 2: Long forward glide [0.15, 0.35]
        elif 0.15 <= phase < 0.35:
            if is_rear:
                # Extended rearward-upward configuration
                foot[0] -= self.step_length * 0.5
                foot[2] += self.glide_extension_height
            else:  # Front legs
                # Extended forward configuration
                foot[0] += self.step_length * 0.8
                progress = (phase - 0.15) / 0.2
                foot[2] += self.rear_bounce_height * (1.0 - 0.5 * progress)
        
        # Phase 3: Front brief contact [0.35, 0.4]
        elif 0.35 <= phase < 0.4:
            if is_front:
                # Stance with compression
                progress = (phase - 0.35) / 0.05
                foot[0] += self.step_length * 0.5
                foot[2] -= 0.03 * np.sin(np.pi * progress)
            else:  # Rear legs
                # Continuing swing forward
                progress = (phase - 0.35) / 0.05
                foot[0] -= self.step_length * 0.3 * (1.0 - progress)
                foot[2] += self.glide_extension_height * (1.0 - 0.3 * progress)
        
        # Phase 4: Front double-bounce [0.4, 0.55]
        elif 0.4 <= phase < 0.55:
            if is_front:
                # Stance with double-bounce oscillation
                progress = (phase - 0.4) / 0.15
                double_bounce_phase = progress * 2.0 * np.pi * 2.0  # Two bounces
                compression = self.double_bounce_amplitude * (1.0 - np.cos(double_bounce_phase))
                foot[2] -= compression
                foot[0] += self.step_length * 0.3
            else:  # Rear legs
                # Swing forward through body
                progress = (phase - 0.4) / 0.15
                foot[0] += self.step_length * progress * 0.6
                foot[2] += self.front_bounce_height * np.sin(np.pi * progress)
        
        # Phase 5: Extended asymmetric glide [0.55, 0.8]
        elif 0.55 <= phase < 0.8:
            progress = (phase - 0.55) / 0.25
            if is_front:
                # Forward-down asymmetric configuration
                foot[0] += self.step_length * 0.4
                foot[2] += self.front_bounce_height * (1.0 - progress)
            else:  # Rear legs
                # Forward swing with asymmetric positioning
                foot[0] += self.step_length * 0.3 * progress
                foot[2] += self.glide_extension_height * (1.0 - progress) * 0.6
        
        # Phase 6: Syncopated landing [0.8, 1.0]
        elif 0.8 <= phase < 1.0:
            if is_fl or is_rr:
                # First diagonal pair (FL+RR): early landing
                if phase < 0.88:
                    # Descending and making contact
                    progress = (phase - 0.8) / 0.08
                    foot[2] -= 0.05 * progress
                else:
                    # Fully in stance
                    foot[0] -= self.step_length * 0.15
                    foot[2] -= 0.05
            else:  # FR and RL
                # Second diagonal pair (FR+RL): delayed landing (syncopation)
                if phase < 0.88:
                    # Still airborne, descending
                    progress = (phase - 0.8) / 0.08
                    foot[2] += self.front_bounce_height * 0.3 * (1.0 - progress)
                else:
                    # Landing phase
                    progress = (phase - 0.88) / 0.12
                    foot[0] -= self.step_length * 0.15 * progress
                    foot[2] -= 0.05 * progress
        
        return foot