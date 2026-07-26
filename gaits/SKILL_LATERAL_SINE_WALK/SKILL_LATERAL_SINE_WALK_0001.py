from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_LATERAL_SINE_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Lateral walking gait to the right with sinusoidal swing height modulation.
    
    - Base moves continuously rightward (positive y) with constant velocity
    - Four legs cycle with 25% phase offsets (90° separation): FL→FR→RL→RR
    - Each leg has ~65% stance duty cycle ensuring 2-3 legs always in contact
    - Swing height modulated sinusoidally per leg to create visual wave pattern
    - Phase offsets: FL=0°, FR=90°, RL=180°, RR=270°
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Complete gait cycle frequency (Hz)
        self.duty = 0.65  # Stance duty cycle (65% stance, 35% swing)
        
        # Lateral stepping parameters
        self.step_length_lateral = 0.15  # Lateral step distance (body +y direction)
        self.base_step_height = 0.06  # Base swing height before sinusoidal modulation
        
        # Sinusoidal height modulation parameters
        self.height_modulation_amplitude = 0.04  # Additional height variation amplitude
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for wave propagation: FL→FR→RL→RR
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL'):
                self.phase_offsets[leg] = 0.0    # 0°
            elif leg.startswith('FR'):
                self.phase_offsets[leg] = 0.25   # 90°
            elif leg.startswith('RL'):
                self.phase_offsets[leg] = 0.5    # 180°
            elif leg.startswith('RR'):
                self.phase_offsets[leg] = 0.75   # 270°
        
        # Sinusoidal height modulation offsets (for visual wave effect)
        self.height_phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL'):
                self.height_phase_offsets[leg] = 0.0         # Peak at phase 0
            elif leg.startswith('FR'):
                self.height_phase_offsets[leg] = 0.5 * np.pi  # Peak at phase 0.25
            elif leg.startswith('RL'):
                self.height_phase_offsets[leg] = np.pi        # Peak at phase 0.5
            elif leg.startswith('RR'):
                self.height_phase_offsets[leg] = 1.5 * np.pi  # Peak at phase 0.75
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Lateral velocity magnitude (constant rightward motion)
        self.lateral_velocity = 0.3  # m/s in +y direction (right)

    def update_base_motion(self, phase, dt):
        """
        Update base with constant lateral velocity to the right.
        No rotation, no forward/backward motion, no vertical drift.
        """
        # Constant lateral velocity in world frame
        self.vel_world = np.array([0.0, self.lateral_velocity, 0.0])
        
        # No angular velocity
        self.omega_world = np.zeros(3)
        
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
        Compute foot position in body frame with:
        1. Lateral stepping motion (rightward during swing, leftward drift during stance)
        2. Sinusoidally modulated swing height creating wave pattern
        """
        # Get leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute sinusoidal height modulation factor
        # This creates the visual wave: peak varies per leg based on their phase offset
        height_modulation = self.height_modulation_amplitude * np.sin(
            2 * np.pi * phase + self.height_phase_offsets[leg_name]
        )
        
        if leg_phase < self.duty:
            # STANCE PHASE
            # Foot drifts leftward in body frame as base moves right in world frame
            stance_progress = leg_phase / self.duty
            foot[1] -= self.step_length_lateral * (stance_progress - 0.5)
            
        else:
            # SWING PHASE
            # Foot lifts and sweeps rightward to new contact position
            swing_progress = (leg_phase - self.duty) / (1.0 - self.duty)
            
            # Lateral sweep: move rightward during swing
            foot[1] += self.step_length_lateral * (swing_progress - 0.5)
            
            # Vertical trajectory: parabolic arc with sinusoidal height modulation
            swing_angle = np.pi * swing_progress
            total_swing_height = self.base_step_height + height_modulation
            foot[2] += total_swing_height * np.sin(swing_angle)
        
        return foot