from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_FIGURE_EIGHT_WEAVE_MotionGenerator(BaseMotionGenerator):
    """
    Figure-eight weave motion with continuous ground contact.
    
    - All four feet remain in stance throughout
    - Base traces figure-eight via alternating yaw rates and constant forward velocity
    - Body leans into turns via differential leg compression (compression-only strategy)
    - Right turn (phase 0.0-0.25): right legs compress, left legs neutral
    - Left turn (phase 0.5-0.75): left legs compress, right legs neutral
    - Center crossings (phase 0.25-0.5, 0.75-1.0): all legs return to neutral
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.4  # ~2.5 seconds per figure-eight cycle
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.vx = 0.4  # Constant forward velocity (m/s)
        self.yaw_rate_amplitude = 1.5  # Yaw rate amplitude (rad/s) for tight turns
        
        # Differential leg compression for body lean into turns
        # Compression-only strategy: inside legs compress, outside legs remain neutral
        # This eliminates ground penetration risk from leg extension
        self.lean_amplitude = 0.02  # Maximum vertical compression (m) for lean effect
        
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
        Compute foot position in BODY frame with compression-only lean strategy.
        
        All feet remain in stance (grounded) throughout.
        
        Body lean created by compressing inside legs while outside legs remain neutral:
        - Right turn (phase 0.0-0.25): right legs compress, left legs neutral
        - Center crossing (phase 0.25-0.5): right legs transition to neutral
        - Left turn (phase 0.5-0.75): left legs compress, right legs neutral
        - Center crossing return (phase 0.75-1.0): left legs transition to neutral
        
        COMPRESSION-ONLY STRATEGY:
        - Inside legs (toward turn center): positive z_offset raises foot toward body (compression)
        - Outside legs (away from turn center): z_offset = 0 (remain at neutral stance height)
        - This eliminates ground penetration risk from leg extension while maintaining lean effect
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left-side or right-side leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compute compression offset based on phase
        # Only inside legs compress; outside legs remain neutral (z_offset = 0)
        
        z_offset = 0.0  # Default: neutral
        
        if phase < 0.25:
            # Right turn: right legs compress (inside), left legs neutral (outside)
            turn_progress = phase / 0.25
            # Smooth sinusoidal modulation
            lean_factor = np.sin(np.pi * turn_progress)
            
            if not is_left_leg:
                # Right leg compresses (positive z_offset raises foot toward body)
                z_offset = self.lean_amplitude * lean_factor
            # Left legs remain neutral (z_offset = 0)
                
        elif phase < 0.5:
            # Center crossing: right legs transition from compressed to neutral
            crossing_progress = (phase - 0.25) / 0.25
            # Smooth transition to neutral
            lean_factor = np.sin(np.pi * (1.0 - crossing_progress / 2.0))
            
            if not is_left_leg:
                z_offset = self.lean_amplitude * lean_factor * (1.0 - crossing_progress)
            # Left legs remain neutral (z_offset = 0)
                
        elif phase < 0.75:
            # Left turn: left legs compress (inside), right legs neutral (outside)
            turn_progress = (phase - 0.5) / 0.25
            # Smooth sinusoidal modulation
            lean_factor = np.sin(np.pi * turn_progress)
            
            if is_left_leg:
                # Left leg compresses (positive z_offset raises foot toward body)
                z_offset = self.lean_amplitude * lean_factor
            # Right legs remain neutral (z_offset = 0)
                
        else:
            # Center crossing return: left legs transition from compressed to neutral
            crossing_progress = (phase - 0.75) / 0.25
            # Smooth transition to neutral
            lean_factor = np.sin(np.pi * (1.0 - crossing_progress / 2.0))
            
            if is_left_leg:
                z_offset = self.lean_amplitude * lean_factor * (1.0 - crossing_progress)
            # Right legs remain neutral (z_offset = 0)
        
        # Apply vertical offset to create lean via compression
        foot[2] += z_offset
        
        return foot