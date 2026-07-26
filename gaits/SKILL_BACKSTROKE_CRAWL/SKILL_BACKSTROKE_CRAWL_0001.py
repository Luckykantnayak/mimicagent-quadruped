from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_BACKSTROKE_CRAWL_MotionGenerator(BaseMotionGenerator):
    """
    Backstroke crawl gait with sequential circular leg sweeps.
    
    - Each leg performs a large circular backstroke arc in sequence: FL → FR → RR → RL
    - Recovery phase: leg lifts and sweeps forward overhead (in body frame)
    - Power phase: leg pushes backward during stance
    - Base moves backward continuously with constant velocity
    - Tripod support maintained throughout (exactly one leg swings at a time)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for large circular sweeps
        
        # Backstroke arc parameters
        self.arc_radius_horizontal = 0.15  # Forward-backward sweep distance
        self.arc_height = 0.12  # Vertical clearance during recovery arc
        self.duty = 0.75  # 75% stance, 25% swing per leg
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Sequential phase offsets: FL → FR → RR → RL (0.25 increments)
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL'):
                self.phase_offsets[leg] = 0.0
            elif leg.startswith('FR'):
                self.phase_offsets[leg] = 0.25
            elif leg.startswith('RR'):
                self.phase_offsets[leg] = 0.5
            elif leg.startswith('RL'):
                self.phase_offsets[leg] = 0.75
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Backward velocity (constant)
        self.vx_backward = -0.3  # Negative for backward motion
        
    def update_base_motion(self, phase, dt):
        """
        Constant backward velocity, no rotation.
        Body remains level throughout the backstroke crawl.
        """
        self.vel_world = np.array([self.vx_backward, 0.0, 0.0])
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
        Compute circular backstroke trajectory for each leg.
        
        Each leg's individual cycle:
        - [0.0, 0.25]: Swing phase - circular recovery arc (lift up, sweep forward)
        - [0.25, 1.0]: Stance phase - power stroke (push backward)
        
        The circular arc during swing resembles a backstroke arm motion:
        foot rises, sweeps forward overhead, then descends to plant forward.
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_phase < (1.0 - self.duty):
            # Swing phase: circular recovery arc (0.0 to 0.25 of leg cycle)
            swing_progress = leg_phase / (1.0 - self.duty)
            
            # Circular trajectory parameterized by angle
            # Start at rear (angle=0), sweep to front (angle=pi)
            arc_angle = np.pi * swing_progress
            
            # Horizontal component: sweep from rear to front
            foot[0] += self.arc_radius_horizontal * (np.cos(arc_angle) + 1.0) - self.arc_radius_horizontal
            
            # Vertical component: rise and fall in smooth arc
            foot[2] += self.arc_height * np.sin(arc_angle)
            
        else:
            # Stance phase: power stroke (0.25 to 1.0 of leg cycle)
            stance_progress = (leg_phase - (1.0 - self.duty)) / self.duty
            
            # Foot moves from forward position to rear position
            # Linear motion during stance as base translates backward
            foot[0] += self.arc_radius_horizontal * (1.0 - 2.0 * stance_progress)
        
        return foot