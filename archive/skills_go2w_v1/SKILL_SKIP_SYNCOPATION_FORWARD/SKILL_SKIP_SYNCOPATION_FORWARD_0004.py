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
        self.step_length = 0.14
        self.rear_bounce_height = 0.055
        self.front_bounce_height = 0.05
        self.glide_extension_height = 0.045  # Reduced from 0.065
        self.double_bounce_amplitude = 0.02
        
        # Base velocity parameters - reduced pitch rates
        self.forward_velocity = 0.8
        self.forward_velocity_glide = 0.6  # Reduced velocity during glide phases
        self.rear_bounce_vz = 0.25
        self.front_bounce_vz = 0.20
        self.rear_bounce_pitch_rate = -0.45  # Reduced from -0.65
        self.front_bounce_pitch_rate = 0.60  # Reduced from 0.85
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
            progress = phase / 0.15
            double_bounce_phase = progress * 2.0 * np.pi * 2.0
            vz = self.rear_bounce_vz * (1.0 + 0.3 * np.sin(double_bounce_phase))
            pitch_rate = self.rear_bounce_pitch_rate
            
        elif 0.15 <= phase < 0.35:
            # Long forward glide: ballistic trajectory with reduced velocity
            vx = self.forward_velocity_glide
            glide_progress = (phase - 0.15) / 0.2
            vz = self.rear_bounce_vz * (1.0 - 2.5 * glide_progress)
            pitch_rate = -0.25 * glide_progress  # Reduced from -0.35
            
        elif 0.35 <= phase < 0.4:
            # Front brief contact: forward with controlled descent
            vx = self.forward_velocity * 0.9
            progress = (phase - 0.35) / 0.05
            vz = -0.25 * progress
            pitch_rate = 0.25  # Reduced from 0.3
            
        elif 0.4 <= phase < 0.55:
            # Front double-bounce: forward and upward, pitch up
            vx = self.forward_velocity
            progress = (phase - 0.4) / 0.15
            double_bounce_phase = progress * 2.0 * np.pi * 2.0
            vz = self.front_bounce_vz * (1.0 + 0.3 * np.sin(double_bounce_phase))
            pitch_rate = self.front_bounce_pitch_rate
            
        elif 0.55 <= phase < 0.8:
            # Extended asymmetric glide: ballistic with pitch deceleration to zero by phase 0.75
            vx = self.forward_velocity_glide
            glide_progress = (phase - 0.55) / 0.25
            vz = self.front_bounce_vz * (1.0 - 2.2 * glide_progress)
            # Decelerate pitch to zero by phase 0.75 (80% through this segment)
            if glide_progress < 0.8:
                pitch_rate = self.front_bounce_pitch_rate * (1.0 - glide_progress / 0.8)
            else:
                pitch_rate = 0.0
            
        elif 0.8 <= phase < 1.0:
            # Syncopated landing: forward with controlled descent
            vx = self.forward_velocity
            landing_progress = (phase - 0.8) / 0.2
            vz = -0.3 * landing_progress
            roll_rate = self.landing_roll_amplitude * np.sin(2.0 * np.pi * landing_progress)
            pitch_rate = -0.4 * landing_progress  # Reduced from -0.5
        
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
        
        # Ground clearance safety margin - increased
        ground_clearance = 0.025  # Increased from 0.015
        
        # Phase 1: Rear double-bounce [0.0, 0.15]
        if 0.0 <= phase < 0.15:
            if is_rear:
                # Stance with minimal compression - bounce primarily from base motion
                progress = phase / 0.15
                double_bounce_phase = progress * 2.0 * np.pi * 2.0
                compression = self.double_bounce_amplitude * 0.5 * (1.0 - np.cos(double_bounce_phase))
                foot[2] = ground_clearance - compression
                # Smooth rearward motion - reduced
                foot[0] -= self.step_length * 0.15 * np.sin(np.pi * progress * 0.5)
            else:  # Front legs
                # Swing forward and upward with smooth curve - reduced extension
                progress = phase / 0.15
                swing_curve = np.sin(np.pi * progress)
                foot[0] += self.step_length * 0.35 * swing_curve  # Reduced from 0.45
                foot[2] += self.rear_bounce_height * swing_curve
        
        # Phase 2: Long forward glide [0.15, 0.35]
        elif 0.15 <= phase < 0.35:
            progress = (phase - 0.15) / 0.2
            smooth_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            if is_rear:
                # Extended rearward-upward configuration - reduced extension
                foot[0] -= self.step_length * 0.15  # Reduced from 0.22
                foot[2] += self.glide_extension_height * smooth_decay * 0.85
            else:  # Front legs
                # Extended forward configuration - reduced extension
                foot[0] += self.step_length * 0.25  # Reduced from 0.38
                foot[2] += self.rear_bounce_height * smooth_decay * 0.9
        
        # Phase 3: Front brief contact [0.35, 0.4]
        elif 0.35 <= phase < 0.4:
            progress = (phase - 0.35) / 0.05
            if is_front:
                # Smooth landing transition to stance - cosine smoothing
                landing_curve = 0.5 * (1.0 - np.cos(np.pi * progress))
                foot[0] += self.step_length * 0.22  # Reduced from 0.32
                foot[2] = ground_clearance + self.rear_bounce_height * 0.25 * (1.0 - landing_curve)
            else:  # Rear legs
                # Continuing forward swing
                smooth_swing = np.sin(np.pi * progress * 0.5)
                foot[0] -= self.step_length * 0.15 * (1.0 - smooth_swing * 0.4)
                foot[2] += self.glide_extension_height * 0.6 * (1.0 - progress * 0.25)
        
        # Phase 4: Front double-bounce [0.4, 0.55]
        elif 0.4 <= phase < 0.55:
            if is_front:
                # Stance with minimal compression
                progress = (phase - 0.4) / 0.15
                double_bounce_phase = progress * 2.0 * np.pi * 2.0
                compression = self.double_bounce_amplitude * 0.5 * (1.0 - np.cos(double_bounce_phase))
                foot[2] = ground_clearance - compression
                # Smooth forward motion during stance - reduced
                foot[0] += self.step_length * 0.18  # Reduced from 0.25
            else:  # Rear legs
                # Swing forward through body - reduced extension
                progress = (phase - 0.4) / 0.15
                swing_curve = np.sin(np.pi * progress)
                foot[0] += self.step_length * 0.25 * progress  # Reduced from 0.35
                foot[2] += self.front_bounce_height * swing_curve
        
        # Phase 5: Extended asymmetric glide [0.55, 0.8]
        elif 0.55 <= phase < 0.8:
            progress = (phase - 0.55) / 0.25
            smooth_descent = 0.5 * (1.0 + np.cos(np.pi * progress))
            if is_front:
                # Forward asymmetric configuration - reduced extension
                foot[0] += self.step_length * 0.18  # Reduced from 0.28
                foot[2] += self.front_bounce_height * smooth_descent * 0.85
            else:  # Rear legs
                # Forward swing with asymmetric positioning - reduced extension
                foot[0] += self.step_length * 0.12  # Reduced from 0.22
                foot[2] += self.glide_extension_height * 0.55 * smooth_descent
        
        # Phase 6: Syncopated landing [0.8, 1.0]
        elif 0.8 <= phase < 1.0:
            if is_fl or is_rr:
                # First diagonal pair (FL+RR): early landing
                if phase < 0.88:
                    # Smooth descent to ground - cosine smoothing, reduced extension
                    progress = (phase - 0.8) / 0.08
                    descent_curve = 0.5 * (1.0 - np.cos(np.pi * progress))
                    foot[0] += self.step_length * 0.06 * (1.0 - progress)  # Reduced from 0.12
                    foot[2] = ground_clearance + self.front_bounce_height * 0.35 * (1.0 - descent_curve)
                else:
                    # In stance
                    progress = (phase - 0.88) / 0.12
                    foot[0] -= self.step_length * 0.06 * progress  # Reduced from 0.08
                    foot[2] = ground_clearance
            else:  # FR and RL
                # Second diagonal pair (FR+RL): delayed landing (syncopation)
                if phase < 0.88:
                    # Still airborne - reduced height
                    progress = (phase - 0.8) / 0.08
                    smooth_hold = 0.5 * (1.0 + np.cos(np.pi * progress))
                    foot[0] += self.step_length * 0.08 * (1.0 - progress * 0.25)  # Reduced from 0.16
                    foot[2] += self.front_bounce_height * 0.28 * smooth_hold  # Reduced from 0.45
                else:
                    # Landing phase with smooth descent - cosine smoothing, reduced extension
                    progress = (phase - 0.88) / 0.12
                    descent_curve = 0.5 * (1.0 - np.cos(np.pi * progress))
                    foot[0] += self.step_length * 0.06 * (1.0 - progress)  # Reduced from 0.08
                    foot[2] = ground_clearance + self.front_bounce_height * 0.28 * (1.0 - descent_curve)
        
        return foot