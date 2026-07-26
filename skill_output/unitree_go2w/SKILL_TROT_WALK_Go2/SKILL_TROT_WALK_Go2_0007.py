from utils.math_utils import *
from motion.base import BaseMotionGenerator
import numpy as np


class SKILL_TROT_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal trot gait for stable forward locomotion.
    
    - FL and RR form diagonal pair 1 (group_1), swing during phase 0.0-0.5
    - FR and RL form diagonal pair 2 (group_2), swing during phase 0.5-1.0
    - Constant forward velocity throughout cycle
    - Double-support transitions at phase boundaries ensure continuous ground contact
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 3.0  # 1 Hz gait frequency for stable walking
        
        # Gait timing parameters
        self.duty_cycle = 0.5  # 50% stance, 50% swing per leg

        # Step geometry
        self.step_length = 0.18  # Forward reach during swing (meters)
        self.step_height = 0.12  # Peak swing foot clearance (meters)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        for leg in self.leg_names:
            width = self.base_feet_pos_body[leg][1]
            print(f"  {leg}: y = {width:.4f} m")
        
        # Phase offsets for diagonal trot coordination
        # FL and RR swing together (phase 0.0-0.5)
        # FR and RL swing together (phase 0.5-1.0)
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0  # Group 1: swing first half
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.phase_offsets[leg] = 0.5  # Group 2: swing second half
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (constant throughout cycle)
        self.vx_forward = 1.  # Forward velocity (m/s) - tunable based on desired speed
        self.vel_world = np.array([self.vx_forward, 0.0, 0.0])
        self.omega_world = np.zeros(3)  # No rotation

    def update_base_motion(self, phase, dt):
        """
        Maintain constant forward velocity with zero angular rates.
        Base moves forward steadily while maintaining level orientation.
        """
        # Constant velocity command throughout entire gait cycle
        self.vel_world = np.array([self.vx_forward, 0.0, 0.0])
        self.omega_world = np.zeros(3)
        
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
        Compute foot trajectory in body frame for given leg and phase.
        
        Stance phase: foot fixed in world, moves rearward in body frame at rate -vx
        Swing phase: foot lifts and swings forward along smooth arc trajectory
        
        Phase timing:
        - Group 1 (FL, RR): swing 0.0-0.5, stance 0.5-1.0
        - Group 2 (FR, RL): stance 0.0-0.5, swing 0.5-1.0
        - Double support: 0.0-0.1, 0.4-0.6, 0.9-1.0
        """
        # Apply diagonal pair phase offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from neutral foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine swing phase boundaries for this leg
        # Swing occurs from leg_phase 0.1 to 0.4 (30% of cycle)
        # Stance covers remaining 70%
        swing_start = 0.
        swing_end = 0.5
        
        if swing_start <= leg_phase < swing_end:
            # SWING PHASE: foot lifts and swings forward
            swing_progress = (leg_phase - swing_start) / (swing_end - swing_start)
            
            # Forward displacement: foot moves from rear to front of step
            # At swing_progress=0: foot at rear (-step_length/2)
            # At swing_progress=1: foot at front (+step_length/2)
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # Vertical arc: smooth lift and descent
            # Peak height at mid-swing (swing_progress=0.5)
            arc_angle = np.pi * swing_progress
            foot[2] += self.step_height * np.sin(arc_angle)
            
        else:
            # STANCE PHASE: foot fixed in world, moves rearward in body frame
            # Compute stance progress from 0 (touchdown) to 1 (liftoff)
            if leg_phase < swing_start:
                # Transition into swing: late stance phase
                stance_progress = (leg_phase + (1.0 - swing_end)) / (swing_start + (1.0 - swing_end))
            else:
                # After swing: early to mid stance
                stance_progress = (leg_phase - swing_end) / (swing_start + (1.0 - swing_end))
            
            # Foot regresses rearward in body frame as base moves forward
            # At stance_progress=0: foot at front (+step_length/2)
            # At stance_progress=1: foot at rear (-step_length/2)
            foot[0] += self.step_length * (0.5 - stance_progress)
        
        return foot