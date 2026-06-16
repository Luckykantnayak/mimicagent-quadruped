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
    
    Base motion uses commanded velocities and angular rates per phase.
    Leg trajectories expressed in BODY frame with phase-dependent swing/stance.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.hop_length = 0.2  # Forward displacement per hop
        self.hop_height = 0.15  # Maximum vertical lift during flight
        self.retraction_height = 0.12  # Leg retraction during swing
        self.compression_depth = 0.06  # Leg compression during prep phase
        
        # Base velocity parameters
        self.vx_liftoff = 0.8  # Forward velocity during liftoff phases
        self.vx_flight = 0.6  # Forward velocity during flight
        self.vz_liftoff = 1.2  # Upward velocity during liftoff
        self.vz_landing = -1.0  # Downward velocity during landing
        
        # Angular velocity parameters
        self.pitch_rate_rear_liftoff = 1.5  # Positive pitch during rear liftoff
        self.pitch_rate_front_liftoff = -1.2  # Negative pitch during front liftoff
        self.pitch_rate_rear_landing = 1.0  # Positive pitch during rear landing
        self.pitch_rate_front_landing = -0.8  # Negative pitch during front landing
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent velocity commands.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        # Phase [0.0, 0.15]: rear_liftoff
        if phase < 0.15:
            vx = self.vx_liftoff
            vz = self.vz_liftoff
            pitch_rate = self.pitch_rate_rear_liftoff
        
        # Phase [0.15, 0.3]: front_liftoff
        elif phase < 0.3:
            vx = self.vx_liftoff
            vz = self.vz_liftoff * 0.7  # Reduced upward velocity
            pitch_rate = self.pitch_rate_front_liftoff
        
        # Phase [0.3, 0.5]: full_flight
        elif phase < 0.5:
            vx = self.vx_flight
            vz = 0.0
            pitch_rate = 0.0
        
        # Phase [0.5, 0.65]: rear_landing
        elif phase < 0.65:
            vx = self.vx_flight * 0.8
            vz = self.vz_landing
            pitch_rate = self.pitch_rate_rear_landing
        
        # Phase [0.65, 0.8]: front_landing
        elif phase < 0.8:
            vx = self.vx_flight * 0.6
            vz = self.vz_landing * 0.5
            pitch_rate = self.pitch_rate_front_landing
        
        # Phase [0.8, 1.0]: compression_prep
        else:
            vx = self.vx_flight * 0.4
            vz = 0.0
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
        Rear leg trajectory: liftoff [0.0, 0.15], swing [0.15, 0.5], landing [0.5, 0.65], stance [0.65, 1.0]
        """
        # Phase [0.0, 0.15]: rear_liftoff (pushoff and retraction)
        if phase < 0.15:
            progress = phase / 0.15
            # Rapid retraction upward
            foot[2] += self.retraction_height * progress
            # Slight rearward motion during pushoff
            foot[0] -= 0.5 * self.hop_length * progress
        
        # Phase [0.15, 0.3]: front_liftoff (continue retraction)
        elif phase < 0.3:
            progress = (phase - 0.15) / 0.15
            # Maintain retracted height, move slightly forward
            foot[2] += self.retraction_height
            foot[0] -= 0.5 * self.hop_length * (1.0 - progress * 0.5)
        
        # Phase [0.3, 0.5]: full_flight (tucked position at apex)
        elif phase < 0.5:
            # Maintain tucked configuration
            foot[2] += self.retraction_height
            foot[0] -= 0.25 * self.hop_length
        
        # Phase [0.5, 0.65]: rear_landing (extension and contact)
        elif phase < 0.65:
            progress = (phase - 0.5) / 0.15
            # Extend downward for landing
            foot[2] += self.retraction_height * (1.0 - progress)
            # Position slightly behind base
            foot[0] -= 0.25 * self.hop_length * (1.0 - progress)
            # Begin compression upon contact
            if progress > 0.3:
                compression_progress = (progress - 0.3) / 0.7
                foot[2] -= self.compression_depth * 0.3 * compression_progress
        
        # Phase [0.65, 0.8]: front_landing (compression continues)
        elif phase < 0.8:
            progress = (phase - 0.65) / 0.15
            # Deeper compression
            foot[2] -= self.compression_depth * (0.3 + 0.4 * progress)
        
        # Phase [0.8, 1.0]: compression_prep (maximum compression)
        else:
            progress = (phase - 0.8) / 0.2
            # Deep compression to store energy
            foot[2] -= self.compression_depth * (0.7 + 0.3 * progress)
        
        return foot

    def _compute_front_leg_trajectory(self, foot, phase):
        """
        Front leg trajectory: stance [0.0, 0.15], liftoff [0.15, 0.3], swing [0.3, 0.65], landing [0.65, 0.8], stance [0.8, 1.0]
        """
        # Phase [0.0, 0.15]: rear_liftoff (front legs in stance, slight extension)
        if phase < 0.15:
            progress = phase / 0.15
            # Slight extension to help push base upward
            foot[2] += 0.02 * np.sin(np.pi * progress)
        
        # Phase [0.15, 0.3]: front_liftoff (rapid retraction)
        elif phase < 0.3:
            progress = (phase - 0.15) / 0.15
            # Rapid retraction upward and slightly rearward
            foot[2] += self.retraction_height * progress
            foot[0] -= 0.3 * self.hop_length * progress
        
        # Phase [0.3, 0.5]: full_flight (tucked at apex)
        elif phase < 0.5:
            # Maintain tucked configuration
            foot[2] += self.retraction_height
            foot[0] -= 0.3 * self.hop_length
        
        # Phase [0.5, 0.65]: rear_landing (front legs still airborne, preparing)
        elif phase < 0.65:
            progress = (phase - 0.5) / 0.15
            # Begin extending forward and downward
            foot[2] += self.retraction_height * (1.0 - 0.5 * progress)
            foot[0] += 0.3 * self.hop_length * progress
        
        # Phase [0.65, 0.8]: front_landing (extension and contact)
        elif phase < 0.8:
            progress = (phase - 0.65) / 0.15
            # Complete extension and land
            foot[2] += self.retraction_height * 0.5 * (1.0 - progress)
            foot[0] += 0.3 * self.hop_length
            # Begin compression upon contact
            if progress > 0.2:
                compression_progress = (progress - 0.2) / 0.8
                foot[2] -= self.compression_depth * 0.5 * compression_progress
        
        # Phase [0.8, 1.0]: compression_prep (deep compression)
        else:
            progress = (phase - 0.8) / 0.2
            # Deep compression
            foot[2] -= self.compression_depth * (0.5 + 0.5 * progress)
        
        return foot