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
        
        # Base foot positions (toe contact) - ensure adequate ground clearance
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            pos = v.copy()
            # Extend toe position downward to ensure adequate ground contact margin
            pos[2] = pos[2] - 0.02  # Additional 2cm downward for toe extension
            self.base_feet_pos_body[k] = pos
        
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
        self.toe_height_variation = 0.008  # Vertical variation for contact modulation (reduced)
        
        # Base motion parameters
        self.forward_velocity = 0.6  # Sustained forward velocity (m/s)
        self.lateral_velocity_amplitude = 0.15  # Lateral velocity modulation amplitude
        self.roll_rate_amplitude = 0.18  # Roll rate amplitude (rad/s) - reduced to prevent excessive accumulation
        self.roll_damping = 0.95  # Damping factor to prevent unbounded roll accumulation
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base using sustained forward velocity with modulated lateral velocity and roll rate.
        
        - Forward velocity (vx): constant positive
        - Lateral velocity (vy): sinusoidal modulation following diagonal weight shifts
        - Roll rate: sinusoidal modulation with damping to prevent unbounded accumulation
        """
        
        # Sustained forward velocity
        vx = self.forward_velocity
        
        # Lateral velocity modulation
        vy = self.lateral_velocity_amplitude * np.sin(2 * np.pi * phase)
        
        # No vertical velocity
        vz = 0.0
        
        # Roll rate modulation following diagonal weight shift
        # Extract current roll angle from quaternion
        current_roll, _, _ = quat_to_euler(self.root_quat)
        
        # Apply damping to prevent unbounded accumulation
        # Roll rate includes both the wave pattern and a restoring component
        target_roll_rate = self.roll_rate_amplitude * np.sin(2 * np.pi * phase)
        restoring_component = -current_roll * 0.5  # Soft restoring force toward zero roll
        roll_rate = target_roll_rate + restoring_component
        
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
        Vertical variation now modulates downward from base to maintain ground contact.
        """
        
        # Get base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Apply phase offset for diagonal coordination
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
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
        
        # Vertical variation for contact modulation
        # Modulate toe extension: more extension (lower z) during loading (inward), less during unloading (outward)
        # Use bipolar modulation centered around base position
        # cos(2π * leg_phase) = 1 at phase 0 (inward, loading) -> maximum downward extension
        # cos(2π * leg_phase) = -1 at phase 0.5 (outward, unloading) -> minimum extension (lift slightly)
        vertical_modulation = -self.toe_height_variation * np.cos(2 * np.pi * leg_phase)
        foot[2] += vertical_modulation  # Negative values extend toe downward, positive lift slightly
        
        return foot


def quat_to_euler(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    quat: [w, x, y, z]
    returns: (roll, pitch, yaw) in radians
    """
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw