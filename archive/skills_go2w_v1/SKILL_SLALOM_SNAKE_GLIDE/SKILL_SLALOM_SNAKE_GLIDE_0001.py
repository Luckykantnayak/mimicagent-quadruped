from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SLALOM_SNAKE_GLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Slalom snake glide skill: continuous forward motion with sinusoidal lateral weaving.
    
    The robot maintains all four feet in continuous ground contact while executing
    a snake-like sinusoidal path. The body curves right, straightens, curves left,
    and straightens again over one complete phase cycle.
    
    Base motion:
    - Constant forward velocity (vx)
    - Sinusoidal lateral velocity (vy) creating the weaving motion
    - Sinusoidal yaw rate coordinated with lateral drift
    
    Leg motion (body frame):
    - All feet remain in stance throughout
    - Legs asymmetrically adjust positions to support body curvature
    - Outside legs extend rearward, inside legs tuck forward during curves
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.  # Slower frequency for smooth, graceful weaving
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base velocity parameters
        self.vx_forward = 0.5  # Constant moderate forward speed
        self.vy_amplitude = 0.3  # Lateral drift amplitude for weaving
        self.yaw_rate_amplitude = 0.8  # Yaw rate amplitude to align with lateral motion
        
        # Leg adjustment parameters (in body frame)
        self.leg_longitudinal_shift = 0.08  # How far legs shift fore/aft during curves
        self.leg_lateral_shift = 0.04  # How much legs tuck inward or extend outward

    def update_base_motion(self, phase, dt):
        """
        Update base motion with constant forward velocity and sinusoidal lateral/yaw motion.
        
        Phase mapping:
        - 0.0-0.25: right curve (vy positive increasing to peak, yaw_rate positive)
        - 0.25-0.5: straightening from right (vy decreasing through zero to negative, yaw_rate decreasing through zero)
        - 0.5-0.75: left curve (vy negative at peak, yaw_rate negative)
        - 0.75-1.0: straightening from left (vy returning to zero, yaw_rate returning to zero)
        """
        
        # Constant forward velocity
        vx = self.vx_forward
        
        # Sinusoidal lateral velocity (peaks at phase 0.125 right, 0.625 left)
        # Using sine wave: sin(2π(phase - 0.25)) gives zero at 0.25, 0.75 and peaks at 0.0, 0.5
        # Shifting to align: sin(2π * phase) gives zero at 0, 0.5, 1.0; peak right at 0.25, left at 0.75
        # We want peak right around 0.125, so use: sin(2π * (phase + 0.125))
        vy = self.vy_amplitude * np.sin(2 * np.pi * phase)
        
        # Sinusoidal yaw rate synchronized with lateral motion
        # Yaw rate should lead slightly or align with lateral velocity to create smooth curving
        # Using same phase relationship as lateral velocity
        yaw_rate = self.yaw_rate_amplitude * np.sin(2 * np.pi * phase)
        
        # No vertical or roll/pitch motion
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
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
        Compute foot position in body frame with asymmetric adjustments during curves.
        
        During right curve (phase 0.0-0.5):
        - Left legs (FL, RL): shift forward and slightly inward (inside of curve)
        - Right legs (FR, RR): shift rearward and outward (outside of curve)
        
        During left curve (phase 0.5-1.0):
        - Right legs (FR, RR): shift forward and slightly inward (inside of curve)
        - Left legs (FL, RL): shift rearward and outward (outside of curve)
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left or right leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Compute smooth longitudinal and lateral adjustments based on phase
        # Use sinusoidal interpolation for smooth transitions
        
        # Curve strength: measures how much we're in a curve vs. neutral
        # Right curve: phase 0.0-0.25 (increasing), 0.25-0.5 (decreasing)
        # Left curve: phase 0.5-0.75 (increasing), 0.75-1.0 (decreasing)
        
        # Right curve factor: positive during right curve, zero at neutral
        # Peaks at phase ~0.125
        right_curve_factor = np.sin(2 * np.pi * phase) if phase < 0.5 else 0.0
        
        # Left curve factor: positive during left curve, zero at neutral
        # Peaks at phase ~0.625
        left_curve_factor = -np.sin(2 * np.pi * phase) if phase >= 0.5 else 0.0
        
        # Smooth blending across full cycle using continuous sinusoid
        curve_phase_continuous = np.sin(2 * np.pi * phase)
        
        # For left legs: positive curve_phase_continuous means right curve (tuck forward/inward)
        #                negative curve_phase_continuous means left curve (extend rearward/outward)
        # For right legs: opposite behavior
        
        if is_left_leg:
            # During right curve (curve_phase > 0): left legs tuck forward and inward
            # During left curve (curve_phase < 0): left legs extend rearward and outward
            longitudinal_adjustment = -self.leg_longitudinal_shift * curve_phase_continuous
            lateral_adjustment = -self.leg_lateral_shift * curve_phase_continuous
            
            foot[0] += longitudinal_adjustment  # x: forward/rearward in body frame
            foot[1] += lateral_adjustment  # y: inward/outward (positive y is left in body frame)
            
        elif is_right_leg:
            # During right curve (curve_phase > 0): right legs extend rearward and outward
            # During left curve (curve_phase < 0): right legs tuck forward and inward
            longitudinal_adjustment = self.leg_longitudinal_shift * curve_phase_continuous
            lateral_adjustment = self.leg_lateral_shift * curve_phase_continuous
            
            foot[0] += longitudinal_adjustment  # x: forward/rearward in body frame
            foot[1] += lateral_adjustment  # y: inward/outward (negative y is right in body frame)
        
        # Z remains at ground level (no lifting)
        # foot[2] unchanged from base position
        
        return foot