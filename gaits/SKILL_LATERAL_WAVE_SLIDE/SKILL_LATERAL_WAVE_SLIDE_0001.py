from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_LATERAL_WAVE_SLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Lateral wave slide gait: Robot slides laterally (left/right) while maintaining
    forward orientation by propagating a sinusoidal body wave from rear to front.
    All four legs maintain continuous sliding ground contact throughout.
    
    - Wave propagates from rear (RL/RR) to front (FL/FR) over one phase cycle
    - Left legs extend outward while right legs compress inward, generating leftward motion
    - Base maintains constant lateral velocity with sinusoidal roll oscillation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Lateral motion parameters
        self.lateral_amplitude = 0.06  # Maximum lateral foot displacement (m)
        self.lateral_velocity = 0.15  # Constant leftward velocity (m/s)
        
        # Roll oscillation parameters
        self.roll_amplitude = 0.12  # Peak roll rate (rad/s), results in ~5-8 deg oscillation
        
        # Wave propagation: rear legs lead front legs by 0.3 phase units
        # Phase offset determines when each leg reaches maximum extension
        self.wave_phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('R'):  # Rear legs (RL, RR)
                self.wave_phase_offsets[leg] = 0.0  # Wave crest at phase 0
            else:  # Front legs (FL, FR)
                self.wave_phase_offsets[leg] = 0.3  # Wave crest at phase 0.3
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state (WORLD frame)
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update_base_motion(self, phase, dt):
        """
        Update base motion with constant lateral velocity and sinusoidal roll oscillation.
        
        - Linear velocity: constant leftward (positive y)
        - Angular velocity: sinusoidal roll rate (negative when left legs extend)
        - Pitch and yaw rates: zero (maintain forward orientation)
        """
        # Constant lateral velocity (leftward in WORLD frame)
        vy = self.lateral_velocity
        
        # Sinusoidal roll rate: negative peak at phase 0 (rear wave crest)
        # Phase shift by π so roll is negative when left legs extend (phase 0)
        roll_rate = self.roll_amplitude * np.sin(2 * np.pi * phase + np.pi)
        
        # Set velocity commands (WORLD frame)
        self.vel_world = np.array([0.0, vy, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
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
        Compute foot position in BODY frame with sinusoidal lateral wave motion.
        
        Wave propagation:
        - Rear legs (RL/RR): wave crest at phase 0
        - Front legs (FL/FR): wave crest at phase 0.3
        
        Left/right asymmetry:
        - Left legs (FL/RL): extend outward (+y) at wave crest
        - Right legs (FR/RR): compress inward (-y) at wave crest
        """
        # Get base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute leg-specific phase with wave propagation offset
        leg_phase = (phase + self.wave_phase_offsets[leg_name]) % 1.0
        
        # Sinusoidal lateral displacement: peak at leg_phase = 0, trough at leg_phase = 0.5
        # cos(2π * leg_phase) gives: 1 at phase 0, -1 at phase 0.5, 1 at phase 1
        wave_value = np.cos(2 * np.pi * leg_phase)
        
        # Apply lateral displacement based on left/right side
        if leg_name.startswith('FL') or leg_name.startswith('RL'):  # Left legs
            # Extend outward (positive y) at wave crest
            lateral_offset = self.lateral_amplitude * wave_value
        else:  # Right legs (FR, RR)
            # Compress inward (negative y) at wave crest (opposite phase)
            lateral_offset = -self.lateral_amplitude * wave_value
        
        # Apply lateral offset to y-coordinate
        foot[1] += lateral_offset
        
        # Z-coordinate remains at ground contact (no vertical motion)
        # X-coordinate unchanged (no forward/backward motion)
        
        return foot