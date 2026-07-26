from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_REVERSE_MOONWALK_MotionGenerator(BaseMotionGenerator):
    """
    Reverse moonwalk gait: robot translates backward while legs perform
    coordinated forward-sliding motions in diagonal pairs.
    
    - All four feet remain in ground contact throughout (sliding gait)
    - Base moves backward continuously
    - Diagonal pairs (FL+RR, FR+RL) alternate forward sliding motions
    - Creates visual illusion of walking forward while moving backward
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Slower frequency for smooth moonwalk illusion
        
        # Base foot positions (neutral stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Moonwalk motion parameters
        self.slide_length = 0.12  # Forward slide distance in body frame
        self.base_backward_speed = 0.15  # Backward velocity magnitude
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Phase timing for diagonal pairs
        # Group 1 (FL, RR): phases 0.0-0.5
        # Group 2 (FR, RL): phases 0.5-1.0
        self.group_1_legs = []
        self.group_2_legs = []
        
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.group_1_legs.append(leg)
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.group_2_legs.append(leg)

    def update_base_motion(self, phase, dt):
        """
        Constant backward velocity throughout the cycle.
        This creates the moonwalk effect when combined with forward leg sliding.
        """
        vx = -self.base_backward_speed  # Negative x = backward motion
        
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on diagonal pair coordination.
        
        Group 1 (FL, RR):
          - phase [0.0, 0.25]: slide forward
          - phase [0.25, 0.5]: slide backward to neutral
          - phase [0.5, 1.0]: hold neutral
        
        Group 2 (FR, RL):
          - phase [0.0, 0.5]: hold neutral
          - phase [0.5, 0.75]: slide forward
          - phase [0.75, 1.0]: slide backward to neutral
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Group 1: FL and RR
        if leg_name in self.group_1_legs:
            if phase < 0.25:
                # Forward slide: 0.0 -> 0.25
                progress = phase / 0.25
                # Smooth forward extension
                offset = self.slide_length * self._smooth_step(progress)
                foot[0] += offset
                
            elif phase < 0.5:
                # Return slide: 0.25 -> 0.5
                progress = (phase - 0.25) / 0.25
                # Smooth backward retraction
                offset = self.slide_length * (1.0 - self._smooth_step(progress))
                foot[0] += offset
                
            # else: phase [0.5, 1.0] - hold neutral (no offset)
        
        # Group 2: FR and RL
        elif leg_name in self.group_2_legs:
            if phase < 0.5:
                # Hold neutral position
                pass
                
            elif phase < 0.75:
                # Forward slide: 0.5 -> 0.75
                progress = (phase - 0.5) / 0.25
                # Smooth forward extension
                offset = self.slide_length * self._smooth_step(progress)
                foot[0] += offset
                
            else:
                # Return slide: 0.75 -> 1.0
                progress = (phase - 0.75) / 0.25
                # Smooth backward retraction
                offset = self.slide_length * (1.0 - self._smooth_step(progress))
                foot[0] += offset
        
        # Keep foot at ground level (stance throughout)
        # No vertical motion - all feet slide along ground
        
        return foot
    
    def _smooth_step(self, t):
        """
        Smoothstep interpolation for smooth transitions.
        Returns value in [0, 1] with zero derivatives at endpoints.
        """
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)