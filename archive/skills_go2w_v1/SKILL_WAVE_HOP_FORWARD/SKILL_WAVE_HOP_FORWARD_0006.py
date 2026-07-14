from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_WAVE_HOP_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Wave hop forward gait with sequential rear-to-front liftoff and landing.
    
    Phase structure:
      [0.0, 0.15]: rear_liftoff - Rear legs push off, front legs in stance
      [0.15, 0.3]: front_liftoff - Front legs push off, all legs airborne
      [0.3, 0.5]: full_flight - Apex of hop with forward translation
      [0.5, 0.65]: rear_landing - Rear legs contact first
      [0.65, 0.8]: front_landing - Front legs contact, all legs in stance
      [0.8, 1.0]: compression_prep - All legs compress for next cycle
    
    Base motion uses commanded velocities with reduced vertical magnitude to stay within safe height envelope.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - tuned to prevent excessive base height
        self.hop_length = 0.18  # Forward displacement per hop
        self.hop_height = 0.10  # Reduced maximum vertical lift during flight
        self.retraction_height = 0.08  # Reduced leg retraction during swing
        self.compression_depth = 0.08  # Increased leg compression during prep phase
        
        # Base velocity parameters - significantly reduced to prevent height violation
        self.vx_liftoff = 0.7  # Forward velocity during liftoff phases
        self.vx_flight = 0.5  # Forward velocity during flight
        self.vz_liftoff = 0.5  # Reduced upward velocity during liftoff
        self.vz_flight_descent = -0.25  # Downward velocity during flight to begin descent
        self.vz_landing = -1.2  # Increased downward velocity during landing
        
        # Angular velocity parameters
        self.pitch_rate_rear_liftoff = 1.2
        self.pitch_rate_front_liftoff = -1.0
        self.pitch_rate_rear_landing = 0.8
        self.pitch_rate_front_landing = -0.6
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent velocity commands with controlled vertical motion.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        # Phase [0.0, 0.15]: rear_liftoff
        if phase < 0.15:
            t = phase / 0.15
            vx = self.vx_liftoff
            vz = self.vz_liftoff * smooth_step(t)
            pitch_rate = self.pitch_rate_rear_liftoff * (1.0 - t)
        
        # Phase [0.15, 0.3]: front_liftoff
        elif phase < 0.3:
            t = (phase - 0.15) / 0.15
            vx = self.vx_liftoff
            vz = self.vz_liftoff * 0.4 * (1.0 - t)  # Reduced and tapering
            pitch_rate = self.pitch_rate_front_liftoff * smooth_step(t)
        
        # Phase [0.3, 0.5]: full_flight - begin descent
        elif phase < 0.5:
            t = (phase - 0.3) / 0.2
            vx = self.vx_flight
            vz = self.vz_flight_descent * smooth_step(t)  # Actively descend during flight
            pitch_rate = 0.0
        
        # Phase [0.5, 0.65]: rear_landing
        elif phase < 0.65:
            t = (phase - 0.5) / 0.15
            vx = self.vx_flight * 0.8
            vz = self.vz_landing * smooth_step(t)
            pitch_rate = self.pitch_rate_rear_landing * (1.0 - t)
        
        # Phase [0.65, 0.8]: front_landing
        elif phase < 0.8:
            t = (phase - 0.65) / 0.15
            vx = self.vx_flight * 0.5
            vz = self.vz_landing * 0.7  # Maintain strong downward velocity
            pitch_rate = self.pitch_rate_front_landing * smooth_step(t)
        
        # Phase [0.8, 1.0]: compression_prep
        else:
            t = (phase - 0.8) / 0.2
            vx = self.vx_flight * 0.3
            vz = -0.2 * t  # Gentle downward motion to compress
            pitch_rate = 0.0
        
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in BODY frame based on phase and leg group.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Rear legs (RL, RR): liftoff at [0.0, 0.15], land at [0.5, 0.65]
        if leg_name.startswith('R'):
            foot = self._compute_rear_leg_trajectory(foot, phase)
        
        # Front legs (FL, FR): liftoff at [0.15, 0.3], land at [0.65, 0.8]
        elif leg_name.startswith('F'):
            foot = self._compute_front_leg_trajectory(foot, phase)
        
        return foot

    def _compute_rear_leg_trajectory(self, foot, phase):
        """
        Rear leg trajectory with smooth transitions and reduced retraction.
        """
        # Phase [0.0, 0.15]: rear_liftoff
        if phase < 0.15:
            t = phase / 0.15
            s = smooth_step(t)
            # Rapid retraction upward with smooth transition
            foot[2] += self.retraction_height * s
            # Slight rearward motion during pushoff
            foot[0] -= 0.4 * self.hop_length * s
        
        # Phase [0.15, 0.3]: front_liftoff
        elif phase < 0.3:
            t = (phase - 0.15) / 0.15
            # Maintain retracted height
            foot[2] += self.retraction_height
            # Transition forward slightly
            foot[0] -= 0.4 * self.hop_length * (1.0 - 0.3 * t)
        
        # Phase [0.3, 0.5]: full_flight
        elif phase < 0.5:
            t = (phase - 0.3) / 0.2
            # Maintain tucked configuration, begin extending for landing
            foot[2] += self.retraction_height * (1.0 - 0.2 * t)
            foot[0] -= 0.28 * self.hop_length
        
        # Phase [0.5, 0.65]: rear_landing
        elif phase < 0.65:
            t = (phase - 0.5) / 0.15
            s = smooth_step(t)
            # Extend downward for landing
            foot[2] += self.retraction_height * 0.8 * (1.0 - s)
            # Position slightly behind base
            foot[0] -= 0.28 * self.hop_length * (1.0 - s * 0.5)
            # Begin compression upon contact
            if t > 0.4:
                comp_t = (t - 0.4) / 0.6
                foot[2] -= self.compression_depth * 0.25 * smooth_step(comp_t)
        
        # Phase [0.65, 0.8]: front_landing
        elif phase < 0.8:
            t = (phase - 0.65) / 0.15
            # Deeper compression
            foot[2] -= self.compression_depth * (0.25 + 0.35 * smooth_step(t))
            foot[0] -= 0.14 * self.hop_length
        
        # Phase [0.8, 1.0]: compression_prep
        else:
            t = (phase - 0.8) / 0.2
            # Maximum compression to lower base
            foot[2] -= self.compression_depth * (0.6 + 0.4 * smooth_step(t))
            foot[0] -= 0.14 * self.hop_length * (1.0 - 0.5 * t)
        
        return foot

    def _compute_front_leg_trajectory(self, foot, phase):
        """
        Front leg trajectory with smooth transitions and reduced retraction.
        """
        # Phase [0.0, 0.15]: rear_liftoff (front legs in stance)
        if phase < 0.15:
            t = phase / 0.15
            # Slight extension to help push base
            foot[2] += 0.015 * np.sin(np.pi * t)
            # Slight rearward shift as base moves forward
            foot[0] -= 0.05 * self.hop_length * t
        
        # Phase [0.15, 0.3]: front_liftoff
        elif phase < 0.3:
            t = (phase - 0.15) / 0.15
            s = smooth_step(t)
            # Rapid retraction upward
            foot[2] += self.retraction_height * s
            # Rearward motion relative to body
            foot[0] -= 0.05 * self.hop_length + 0.25 * self.hop_length * s
        
        # Phase [0.3, 0.5]: full_flight
        elif phase < 0.5:
            t = (phase - 0.3) / 0.2
            # Maintain tucked, begin slight forward extension
            foot[2] += self.retraction_height * (1.0 - 0.1 * t)
            foot[0] -= 0.3 * self.hop_length * (1.0 - 0.3 * t)
        
        # Phase [0.5, 0.65]: rear_landing (front still airborne)
        elif phase < 0.65:
            t = (phase - 0.5) / 0.15
            s = smooth_step(t)
            # Begin extending forward and downward
            foot[2] += self.retraction_height * 0.9 * (1.0 - 0.6 * s)
            foot[0] += 0.21 * self.hop_length * (0.3 + 0.7 * s) - 0.3 * self.hop_length
        
        # Phase [0.65, 0.8]: front_landing
        elif phase < 0.8:
            t = (phase - 0.65) / 0.15
            s = smooth_step(t)
            # Complete extension and land
            foot[2] += self.retraction_height * 0.36 * (1.0 - s)
            foot[0] += 0.21 * self.hop_length
            # Begin compression upon contact
            if t > 0.3:
                comp_t = (t - 0.3) / 0.7
                foot[2] -= self.compression_depth * 0.4 * smooth_step(comp_t)
        
        # Phase [0.8, 1.0]: compression_prep
        else:
            t = (phase - 0.8) / 0.2
            # Deep compression
            foot[2] -= self.compression_depth * (0.4 + 0.6 * smooth_step(t))
            foot[0] += 0.21 * self.hop_length * (1.0 - 0.3 * t)
        
        return foot