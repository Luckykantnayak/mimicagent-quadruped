from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_INVERTED_WAVE_HANDSTAND_MotionGenerator(BaseMotionGenerator):
    """
    Inverted handstand with rear leg lateral wave carving motion.
    
    - Front legs (FL, FR) remain static in body frame, providing handstand support
    - Rear legs (RL, RR) execute anti-phase lateral waves while maintaining contact
    - Base executes cyclic roll and yaw oscillations synchronized with rear leg waves
    - Lateral velocity accumulates through carving dynamics
    - Pitch maintained near constant to hold inverted posture
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for controlled inverted motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Identify leg groups
        self.front_legs = [leg for leg in leg_names if leg.startswith('F')]
        self.rear_legs = [leg for leg in leg_names if leg.startswith('R')]
        
        # Rear leg lateral wave parameters
        self.rear_lateral_amplitude = 0.12  # Lateral sweep amplitude in body frame (meters)
        
        # Base angular velocity amplitudes (conservative to maintain handstand stability)
        self.roll_rate_amp = 0.4  # rad/s
        self.yaw_rate_amp = 0.5   # rad/s
        self.pitch_rate = 0.0     # Maintain constant pitch for inversion
        
        # Base lateral velocity amplitude
        self.vy_amp = 0.3  # m/s lateral drift
        
        # Time tracking
        self.t = 0.0
        
        # Base state (world frame)
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base using cyclic roll, yaw, and lateral velocity synchronized with wave.
        
        Phase cycle:
        - [0.0, 0.25]: Left carve initiation (negative roll rate, positive yaw rate)
        - [0.25, 0.5]: Left carve peak then transition (roll/yaw rates reduce)
        - [0.5, 0.75]: Right carve initiation (positive roll rate, negative yaw rate)
        - [0.75, 1.0]: Right carve peak then return (roll/yaw rates reduce to close cycle)
        """
        
        # Roll rate: sinusoidal oscillation with phase offset for left-right tilt
        # Negative in first half (left tilt), positive in second half (right tilt)
        roll_rate = -self.roll_rate_amp * np.sin(2 * np.pi * phase)
        
        # Yaw rate: synchronized with roll for carving coordination
        # Positive during left carve, negative during right carve
        yaw_rate = self.yaw_rate_amp * np.sin(2 * np.pi * phase)
        
        # Lateral velocity: oscillates to create carving drift
        # Negative (leftward) in first half, positive (rightward) in second half
        vy = -self.vy_amp * np.sin(2 * np.pi * phase)
        
        # No forward/backward motion (vx = 0), no vertical motion (vz = 0)
        self.vel_world = np.array([0.0, vy, 0.0])
        
        # Angular velocity: roll and yaw modulation, pitch held constant
        self.omega_world = np.array([roll_rate, self.pitch_rate, yaw_rate])
        
        # Integrate pose in world frame
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        
        Front legs (FL, FR): Static stance throughout cycle
        Rear legs (RL, RR): Anti-phase lateral wave motion
        
        RL wave: inward [0.0-0.25], neutral to outward [0.25-0.5], 
                 outward [0.5-0.75], neutral to inward [0.75-1.0]
        RR wave: outward [0.0-0.25], neutral to inward [0.25-0.5],
                 inward [0.5-0.75], neutral to outward [0.75-1.0]
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Front legs: static stance (no modification)
        if leg_name.startswith('F'):
            return foot
        
        # Rear legs: lateral wave motion
        if leg_name.startswith('R'):
            # Determine leg-specific phase offset for anti-phase coordination
            if 'L' in leg_name:  # RL
                # RL starts moving inward (negative y offset)
                lateral_offset = -self.rear_lateral_amplitude * np.cos(2 * np.pi * phase)
            else:  # RR
                # RR starts moving outward (positive y offset) - anti-phase with RL
                lateral_offset = self.rear_lateral_amplitude * np.cos(2 * np.pi * phase)
            
            # Apply lateral sweep in body frame y-axis
            foot[1] += lateral_offset
        
        return foot