from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_FIGURE_EIGHT_WEAVE_MotionGenerator(BaseMotionGenerator):
    """
    Figure-eight weave motion with continuous ground contact.
    
    - All four feet remain in stance throughout
    - Base traces figure-eight via alternating yaw rates and constant forward velocity
    - Body leans into turns via differential leg extension
    - Right turn (phase 0.0-0.25): right legs compress, left legs extend
    - Left turn (phase 0.5-0.75): left legs compress, right legs extend
    - Center crossings (phase 0.25-0.5, 0.75-1.0): legs equalize
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.4  # ~2.5 seconds per figure-eight cycle
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.vx = 0.4  # Constant forward velocity (m/s)
        self.yaw_rate_amplitude = 1.5  # Yaw rate amplitude (rad/s) for tight turns
        
        # Differential leg extension for body lean into turns
        # Reduced amplitude to avoid joint limits and ground penetration
        self.lean_amplitude = 0.04  # Maximum vertical displacement (m) for lean effect
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (updated per step)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity and phase-dependent yaw rate.
        
        Yaw rate profile:
        - Phase 0.0-0.25: negative (right turn)
        - Phase 0.25-0.5: transition negative -> positive (center crossing)
        - Phase 0.5-0.75: positive (left turn)
        - Phase 0.75-1.0: transition positive -> negative (center crossing return)
        
        Uses sinusoidal yaw rate: yaw_rate = A * sin(2π * (phase - 0.25))
        - Zero crossings at phase 0.25 and 0.75
        - Negative peak at phase 0.125
        - Positive peak at phase 0.625
        """
        
        # Constant forward velocity in world frame
        vx = self.vx
        
        # Sinusoidal yaw rate with zero crossings at phase 0.25 and 0.75
        # Shift phase by -0.25 so that sin(0) occurs at phase 0.25
        yaw_rate = self.yaw_rate_amplitude * np.sin(2 * np.pi * (phase - 0.25))
        
        # Set velocity commands
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate base pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in BODY frame with differential leg extension.
        
        All feet remain in stance (grounded) throughout.
        
        Leg extension creates body lean:
        - Right turn (phase 0.0-0.25): right legs compress (shorten), left legs extend (lengthen)
        - Center crossing (phase 0.25-0.5): legs equalize
        - Left turn (phase 0.5-0.75): left legs compress (shorten), right legs extend (lengthen)
        - Center crossing return (phase 0.75-1.0): legs equalize
        
        CORRECTED SIGN CONVENTION:
        In body frame, feet are at negative z (below body center).
        - To COMPRESS a leg (raise foot toward body): apply POSITIVE z_offset
        - To EXTEND a leg (lower foot away from body): apply NEGATIVE z_offset
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left-side or right-side leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Compute differential extension based on phase
        # Left legs: extend (lengthen) during right turn, compress (shorten) during left turn
        # Right legs: compress (shorten) during right turn, extend (lengthen) during left turn
        
        if phase < 0.25:
            # Right turn: right legs compress, left legs extend
            # Progress through right turn
            turn_progress = phase / 0.25
            # Smooth sinusoidal modulation
            lean_factor = np.sin(np.pi * turn_progress)
            
            if is_left_leg:
                # Left leg extends (negative z_offset pushes foot down)
                z_offset = -self.lean_amplitude * lean_factor
            else:
                # Right leg compresses (positive z_offset raises foot toward body)
                z_offset = self.lean_amplitude * lean_factor
                
        elif phase < 0.5:
            # Center crossing: transition to neutral
            # Progress through center crossing
            crossing_progress = (phase - 0.25) / 0.25
            # Smooth transition to neutral (lean factor decays to zero)
            lean_factor = np.sin(np.pi * (1.0 - crossing_progress / 2.0))
            
            if is_left_leg:
                z_offset = -self.lean_amplitude * lean_factor * (1.0 - crossing_progress)
            else:
                z_offset = self.lean_amplitude * lean_factor * (1.0 - crossing_progress)
                
        elif phase < 0.75:
            # Left turn: left legs compress, right legs extend
            # Progress through left turn
            turn_progress = (phase - 0.5) / 0.25
            # Smooth sinusoidal modulation
            lean_factor = np.sin(np.pi * turn_progress)
            
            if is_left_leg:
                # Left leg compresses (positive z_offset raises foot toward body)
                z_offset = self.lean_amplitude * lean_factor
            else:
                # Right leg extends (negative z_offset pushes foot down)
                z_offset = -self.lean_amplitude * lean_factor
                
        else:
            # Center crossing return: transition to neutral
            # Progress through center crossing return
            crossing_progress = (phase - 0.75) / 0.25
            # Smooth transition to neutral (lean factor decays to zero)
            lean_factor = np.sin(np.pi * (1.0 - crossing_progress / 2.0))
            
            if is_left_leg:
                z_offset = self.lean_amplitude * lean_factor * (1.0 - crossing_progress)
            else:
                z_offset = -self.lean_amplitude * lean_factor * (1.0 - crossing_progress)
        
        # Apply vertical offset to create lean
        foot[2] += z_offset
        
        return foot