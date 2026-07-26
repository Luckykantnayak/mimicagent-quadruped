from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_WAVE_TOE_SURF_MotionGenerator(BaseMotionGenerator):
    """
    Wave Toe Surf: Continuous forward motion on toes with diagonal leg waves.
    
    - All four legs maintain continuous toe contact throughout motion
    - Diagonal pairs (FL+RR vs FR+RL) alternate inward/outward wave patterns
    - Base maintains sustained forward velocity with modulated lateral velocity
    - Roll rate follows diagonal weight shift patterns
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Motion frequency (Hz)
        
        # Base foot positions (toe contact)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Diagonal group definitions
        self.group_1 = []  # FL, RR
        self.group_2 = []  # FR, RL
        
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.group_1.append(leg)
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.group_2.append(leg)
        
        # Phase offsets for diagonal coordination
        # Group 1 (FL+RR) starts inward at phase 0
        # Group 2 (FR+RL) starts outward at phase 0 (0.5 offset)
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.phase_offsets[leg] = 0.5
        
        # Wave motion parameters
        self.lateral_wave_amplitude = 0.08  # Lateral wave amplitude (m)
        self.toe_height_variation = 0.01  # Small vertical variation for contact modulation
        
        # Base motion parameters
        self.forward_velocity = 0.6  # Sustained forward velocity (m/s)
        self.lateral_velocity_amplitude = 0.15  # Lateral velocity modulation amplitude
        self.roll_rate_amplitude = 0.3  # Roll rate amplitude (rad/s)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base using sustained forward velocity with modulated lateral velocity and roll rate.
        
        - Forward velocity (vx): constant positive
        - Lateral velocity (vy): sinusoidal modulation following diagonal weight shifts
        - Roll rate: sinusoidal modulation following diagonal wave pattern
        """
        
        # Sustained forward velocity
        vx = self.forward_velocity
        
        # Lateral velocity modulation
        # Positive when group_1 (FL+RR) waves inward (phase 0-0.25, 0.75-1.0)
        # Negative when group_2 (FR+RL) waves inward (phase 0.25-0.75)
        vy = self.lateral_velocity_amplitude * np.sin(2 * np.pi * phase)
        
        # No vertical velocity
        vz = 0.0
        
        # Roll rate modulation following diagonal weight shift
        # Positive when group_1 inward (right diagonal loaded)
        # Negative when group_2 inward (left diagonal loaded)
        roll_rate = self.roll_rate_amplitude * np.sin(2 * np.pi * phase)
        
        # No pitch or yaw rate
        pitch_rate = 0.0
        yaw_rate = 0.0
        
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
        Compute foot position in body frame with continuous toe contact and lateral wave motion.
        
        Wave pattern:
        - Group 1 (FL+RR): inward at phase 0-0.25, outward at phase 0.25-0.75, return at 0.75-1.0
        - Group 2 (FR+RL): outward at phase 0-0.25, inward at phase 0.25-0.75, return at 0.75-1.0
        
        All feet maintain toe contact throughout with small vertical variations.
        """
        
        # Get base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Apply phase offset for diagonal coordination
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Determine lateral wave direction based on group membership
        # For group_1 (FL+RR): leg_phase 0 = inward, 0.5 = outward
        # For group_2 (FR+RL): leg_phase 0 = outward, 0.5 = inward (due to 0.5 offset)
        
        # Smooth sinusoidal lateral wave
        # Negative = inward (toward body centerline)
        # Positive = outward (away from body centerline)
        lateral_offset = -self.lateral_wave_amplitude * np.cos(2 * np.pi * leg_phase)
        
        # Determine sign based on leg position (left legs positive y, right legs negative y)
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            # Left legs: positive y in body frame
            foot[1] += lateral_offset
        elif leg_name.startswith('FR') or leg_name.startswith('RR'):
            # Right legs: negative y in body frame
            foot[1] -= lateral_offset
        
        # Small vertical variation to modulate contact pressure
        # Maximum toe extension when moving inward (loading phase)
        # Slightly less extension when moving outward (unloading phase)
        vertical_variation = self.toe_height_variation * (0.5 + 0.5 * np.cos(2 * np.pi * leg_phase))
        foot[2] += vertical_variation
        
        return foot