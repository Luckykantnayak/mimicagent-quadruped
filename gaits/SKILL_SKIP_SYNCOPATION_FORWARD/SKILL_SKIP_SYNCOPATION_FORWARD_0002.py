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
        
        # Motion parameters - reduced for joint limit safety
        self.step_length = 0.16
        self.rear_bounce_height = 0.07
        self.front_bounce_height = 0.06
        self.glide_extension_height = 0.09
        self.double_bounce_amplitude = 0.02  # Visual bounce through base motion primarily
        
        # Base velocity parameters - reduced vertical velocities
        self.forward_velocity = 0.8
        self.rear_bounce_vz = 0.25
        self.front_bounce_vz = 0.20
        self.rear_bounce_pitch_rate = -1.0
        self.front_bounce_pitch_rate = 1.2
        self.landing_roll_amplitude = 0.6
        
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
            # Oscillating vertical velocity for bounce effect
            progress = phase / 0.15
            double_bounce_phase = progress * 2.0 * np.pi * 2.0
            vz = self.rear_bounce_vz * (1.0 + 0.3 * np.sin(double_bounce_phase))
            pitch_rate = self.rear_bounce_pitch_rate
            
        elif 0.15 <= phase < 0.35:
            # Long forward glide: ballistic trajectory
            vx = self.forward_velocity
            glide_progress = (phase - 0.15) / 0.2
            vz = self.rear_bounce_vz * (1.0 - 2.5 * glide_progress)
            pitch_rate = -0.5 * glide_progress
            
        elif 0.35 <= phase < 0.4:
            # Front brief contact: forward with controlled descent
            vx = self.forward_velocity * 0.9
            progress = (phase - 0.35) / 0.05
            vz = -0.25 * progress
            pitch_rate = 0.3
            
        elif 0.4 <= phase < 0.55:
            # Front double-bounce: forward and upward, pitch up
            vx = self.forward_velocity
            progress = (phase - 0.4) / 0.15
            double_bounce_phase = progress * 2.0 * np.pi * 2.0
            vz = self.front_bounce_vz * (1.0 + 0.3 * np.sin(double_bounce_phase))
            pitch_rate = self.front_bounce_pitch_rate
            
        elif 0.55 <= phase < 0.8:
            # Extended asymmetric glide: ballistic with pitch adjustment
            vx = self.forward_velocity
            glide_progress = (phase - 0.55) / 0.25
            vz = self.front_bounce_vz * (1.0 - 2.2 * glide_progress)
            pitch_rate = self.front_bounce_pitch_rate * (1.0 - glide_progress)
            
        elif 0.8 <= phase < 1.0:
            # Syncopated landing: forward with controlled descent
            vx = self.forward_velocity
            landing_progress = (phase - 0.8) / 0.2
            vz = -0.3 * landing_progress
            roll_rate = self.landing_roll_amplitude * np.sin(2.0 * np.pi * landing_progress)
            pitch_rate = -0.5 * landing_progress
        
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
        
        # Ground clearance safety margin
        ground_clearance = 0.015
        
        # Phase 1: Rear double-bounce [0.0, 0.15]
        if 0.0 <= phase < 0.15:
            if is_rear:
                # Stance with minimal compression - bounce primarily from base motion
                progress = phase / 0.15
                double_bounce_phase = progress * 2.0 * np.pi * 2.0
                # Small compression that maintains ground contact
                compression = self.double_bounce_amplitude * 0.5 * (1.0 - np.cos(double_bounce_phase))
                foot[2] = ground_clearance - compression
                # Smooth rearward motion
                foot[0] -= self.step_length * 0.25 * np.sin(np.pi * progress * 0.5)
            else:  # Front legs
                # Swing forward and upward with smooth curve
                progress = phase / 0.15
                swing_curve = np.sin(np.pi * progress * 0.5)
                foot[0] += self.step_length * 0.6 * swing_curve
                foot[2] += self.rear_bounce_height * swing_curve
        
        # Phase 2: Long forward glide [0.15, 0.35]
        elif 0.15 <= phase < 0.35:
            progress = (phase - 0.15) / 0.2
            if is_rear:
                # Extended rearward-upward configuration
                foot[0] -= self.step_length * 0.35
                foot[2] += self.glide_extension_height * (1.0 - 0.3 * progress)
            else:  # Front legs
                # Extended forward configuration
                foot[0] += self.step_length * 0.55
                foot[2] += self.rear_bounce_height * (1.0 - 0.4 * progress)
        
        # Phase 3: Front brief contact [0.35, 0.4]
        elif 0.35 <= phase < 0.4:
            progress = (phase - 0.35) / 0.05
            if is_front:
                # Smooth landing transition to stance
                landing_curve = 1.0 - (1.0 - progress) ** 2
                foot[0] += self.step_length * 0.4
                foot[2] = ground_clearance + self.rear_bounce_height * 0.3 * (1.0 - landing_curve)
            else:  # Rear legs
                # Continuing forward swing
                foot[0] -= self.step_length * 0.35 * (1.0 - progress * 0.5)
                foot[2] += self.glide_extension_height * 0.7 * (1.0 - progress * 0.3)
        
        # Phase 4: Front double-bounce [0.4, 0.55]
        elif 0.4 <= phase < 0.55:
            if is_front:
                # Stance with minimal compression
                progress = (phase - 0.4) / 0.15
                double_bounce_phase = progress * 2.0 * np.pi * 2.0
                compression = self.double_bounce_amplitude * 0.5 * (1.0 - np.cos(double_bounce_phase))
                foot[2] = ground_clearance - compression
                # Smooth forward motion during stance
                foot[0] += self.step_length * 0.3
            else:  # Rear legs
                # Swing forward through body
                progress = (phase - 0.4) / 0.15
                swing_curve = np.sin(np.pi * progress)
                foot[0] += self.step_length * 0.5 * progress
                foot[2] += self.front_bounce_height * swing_curve
        
        # Phase 5: Extended asymmetric glide [0.55, 0.8]
        elif 0.55 <= phase < 0.8:
            progress = (phase - 0.55) / 0.25
            if is_front:
                # Forward asymmetric configuration
                foot[0] += self.step_length * 0.35
                foot[2] += self.front_bounce_height * (1.0 - progress * 0.8)
            else:  # Rear legs
                # Forward swing with asymmetric positioning
                foot[0] += self.step_length * 0.3
                foot[2] += self.glide_extension_height * 0.6 * (1.0 - progress)
        
        # Phase 6: Syncopated landing [0.8, 1.0]
        elif 0.8 <= phase < 1.0:
            if is_fl or is_rr:
                # First diagonal pair (FL+RR): early landing
                if phase < 0.88:
                    # Smooth descent to ground
                    progress = (phase - 0.8) / 0.08
                    descent_curve = progress ** 0.5
                    foot[0] += self.step_length * 0.15 * (1.0 - progress)
                    foot[2] = ground_clearance + self.front_bounce_height * 0.4 * (1.0 - descent_curve)
                else:
                    # In stance
                    progress = (phase - 0.88) / 0.12
                    foot[0] -= self.step_length * 0.1 * progress
                    foot[2] = ground_clearance
            else:  # FR and RL
                # Second diagonal pair (FR+RL): delayed landing (syncopation)
                if phase < 0.88:
                    # Still airborne
                    progress = (phase - 0.8) / 0.08
                    foot[0] += self.step_length * 0.2 * (1.0 - progress * 0.3)
                    foot[2] += self.front_bounce_height * 0.5 * (1.0 - progress * 0.6)
                else:
                    # Landing phase with smooth descent
                    progress = (phase - 0.88) / 0.12
                    descent_curve = progress ** 0.7
                    foot[0] += self.step_length * 0.1 * (1.0 - progress)
                    foot[2] = ground_clearance + self.front_bounce_height * 0.3 * (1.0 - descent_curve)
        
        return foot