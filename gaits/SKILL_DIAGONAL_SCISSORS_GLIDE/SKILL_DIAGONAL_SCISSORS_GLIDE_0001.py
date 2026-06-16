from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_DIAGONAL_SCISSORS_GLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal Scissors Glide locomotion skill.
    
    Motion pattern:
    - Diagonal leg pairs (FL/RR and FR/RL) alternate between extended and crossed positions
    - Extended legs provide wide stance stability
    - Crossed legs generate diagonal push impulses
    - Base moves forward-right with velocity modulation (push-glide-push rhythm)
    - All four legs maintain ground contact throughout cycle
    
    Phase structure:
    [0.0-0.2]: FL/RR extend, FR/RL cross inward (setup)
    [0.2-0.4]: FR/RL push from crossed position (first push)
    [0.4-0.6]: All legs transition, base glides on momentum
    [0.6-0.8]: FR/RL extend, FL/RR cross inward (setup reversed)
    [0.8-1.0]: FL/RR push from crossed position (second push)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions (neutral stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Scissor motion parameters
        self.extension_offset_lateral = 0.12  # How far legs extend outward
        self.extension_offset_longitudinal = 0.08  # Forward/backward extension
        self.crossing_offset_lateral = 0.06  # How far legs cross inward
        self.crossing_offset_longitudinal = 0.03  # Slight forward shift when crossing
        
        # Base velocity parameters
        self.vx_base = 0.6  # Base forward velocity (m/s)
        self.vy_base = 0.6  # Base rightward velocity (m/s) - matched for 45° diagonal
        self.vx_push_multiplier = 1.8  # Velocity increase during push phases
        self.vy_push_multiplier = 1.8
        self.vx_glide_multiplier = 0.5  # Velocity reduction during glide
        self.vy_glide_multiplier = 0.5
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Identify leg sides for motion logic
        self.front_left = [leg for leg in leg_names if leg.startswith('FL')][0]
        self.front_right = [leg for leg in leg_names if leg.startswith('FR')][0]
        self.rear_left = [leg for leg in leg_names if leg.startswith('RL')][0]
        self.rear_right = [leg for leg in leg_names if leg.startswith('RR')][0]

    def update_base_motion(self, phase, dt):
        """
        Update base velocity based on phase to create push-glide rhythm.
        
        Velocity profile:
        - Setup phases (0.0-0.2, 0.6-0.8): moderate diagonal velocity
        - Push phases (0.2-0.4, 0.8-1.0): increased diagonal velocity
        - Glide phase (0.4-0.6): reduced velocity (momentum coasting)
        """
        
        # Determine velocity multipliers based on phase
        if 0.0 <= phase < 0.2:
            # First setup: moderate velocity
            vx_mult = 1.0
            vy_mult = 1.0
        elif 0.2 <= phase < 0.4:
            # First push: increased velocity with smooth ramp
            progress = (phase - 0.2) / 0.2
            vx_mult = 1.0 + (self.vx_push_multiplier - 1.0) * (0.5 - 0.5 * np.cos(np.pi * progress))
            vy_mult = 1.0 + (self.vy_push_multiplier - 1.0) * (0.5 - 0.5 * np.cos(np.pi * progress))
        elif 0.4 <= phase < 0.6:
            # Glide: reduced velocity with decay
            progress = (phase - 0.4) / 0.2
            vx_mult = self.vx_push_multiplier - (self.vx_push_multiplier - self.vx_glide_multiplier) * progress
            vy_mult = self.vy_push_multiplier - (self.vy_push_multiplier - self.vy_glide_multiplier) * progress
        elif 0.6 <= phase < 0.8:
            # Second setup: moderate velocity recovery
            progress = (phase - 0.6) / 0.2
            vx_mult = self.vx_glide_multiplier + (1.0 - self.vx_glide_multiplier) * progress
            vy_mult = self.vy_glide_multiplier + (1.0 - self.vy_glide_multiplier) * progress
        else:  # 0.8 <= phase < 1.0
            # Second push: increased velocity
            progress = (phase - 0.8) / 0.2
            vx_mult = 1.0 + (self.vx_push_multiplier - 1.0) * (0.5 - 0.5 * np.cos(np.pi * progress))
            vy_mult = 1.0 + (self.vy_push_multiplier - 1.0) * (0.5 - 0.5 * np.cos(np.pi * progress))
        
        vx = self.vx_base * vx_mult
        vy = self.vy_base * vy_mult
        
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, 0.0])  # No rotation - stable orientation
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on scissoring pattern.
        
        Group 1 (FL/RR): Extend [0.0-0.4], Cross [0.6-1.0]
        Group 2 (FR/RL): Cross [0.0-0.4], Extend [0.6-1.0]
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this leg is in group 1 (FL/RR) or group 2 (FR/RL)
        is_group1 = leg_name.startswith('FL') or leg_name.startswith('RR')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        if is_group1:
            # FL/RR: extend [0.0-0.4], transition [0.4-0.6], cross [0.6-1.0]
            if phase < 0.4:
                # Extending phase
                progress = min(phase / 0.2, 1.0)  # Reach full extension by phase 0.2
                lateral_offset = self.extension_offset_lateral * progress
                longitudinal_offset = self.extension_offset_longitudinal * progress
            elif phase < 0.6:
                # Transition from extended to crossed
                progress = (phase - 0.4) / 0.2
                lateral_offset = self.extension_offset_lateral * (1.0 - progress) - self.crossing_offset_lateral * progress
                longitudinal_offset = self.extension_offset_longitudinal * (1.0 - progress) + self.crossing_offset_longitudinal * progress
            else:
                # Crossing phase (0.6-1.0)
                if phase < 0.8:
                    progress = (phase - 0.6) / 0.2
                else:
                    progress = 1.0  # Hold crossed position during push
                lateral_offset = -self.crossing_offset_lateral * progress
                longitudinal_offset = self.crossing_offset_longitudinal * progress
        else:
            # FR/RL: cross [0.0-0.4], transition [0.4-0.6], extend [0.6-1.0]
            if phase < 0.2:
                # Moving to crossed position
                progress = phase / 0.2
                lateral_offset = -self.crossing_offset_lateral * progress
                longitudinal_offset = self.crossing_offset_longitudinal * progress
            elif phase < 0.4:
                # Hold crossed position during push
                lateral_offset = -self.crossing_offset_lateral
                longitudinal_offset = self.crossing_offset_longitudinal
            elif phase < 0.6:
                # Transition from crossed to extended
                progress = (phase - 0.4) / 0.2
                lateral_offset = -self.crossing_offset_lateral * (1.0 - progress) + self.extension_offset_lateral * progress
                longitudinal_offset = self.crossing_offset_longitudinal * (1.0 - progress) + self.extension_offset_longitudinal * progress
            else:
                # Extended phase (0.6-1.0)
                progress = min((phase - 0.6) / 0.2, 1.0)
                lateral_offset = self.extension_offset_lateral * progress
                longitudinal_offset = self.extension_offset_longitudinal * progress
        
        # Apply offsets with correct signs based on leg position
        lateral_sign = -1.0 if is_left else 1.0
        longitudinal_sign = 1.0 if is_front else -1.0
        
        foot = base_pos.copy()
        foot[0] += longitudinal_offset * longitudinal_sign  # x: forward/backward
        foot[1] += lateral_offset * lateral_sign  # y: left/right
        
        # Keep z (height) constant - all legs maintain ground contact
        
        return foot