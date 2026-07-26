from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_WINDMILL_ROTATION_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Windmill rotation walk gait.
    
    - Right-side legs (FR, RR) execute synchronized windmill circular trajectories
      from phase 0.0-0.5 (swing), then stance from 0.5-1.0
    - Left-side legs (FL, RL) are offset by 0.5 phase, executing stance from 0.0-0.5
      and windmill swing from 0.5-1.0
    - Base moves forward with modulated velocity aligned to stance phases
    - Large vertical circular trajectories create distinctive windmill visual effect
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.6
        
        # Reduced windmill trajectory parameters to respect joint limits
        self.windmill_radius_vertical = 0.10  # Reduced from 0.18m
        self.windmill_radius_horizontal = 0.07  # Reduced from 0.12m
        self.vertical_offset = 0.04  # Minimum clearance above neutral stance
        self.step_height = self.windmill_radius_vertical
        
        # Base foot positions (neutral stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for side-alternating windmill pattern
        self.phase_offsets = {}
        for leg in leg_names:
            if 'R' in leg and 'F' in leg:  # FR
                self.phase_offsets[leg] = 0.0
            elif 'R' in leg and 'R' in leg and 'F' not in leg:  # RR
                self.phase_offsets[leg] = 0.0
            elif 'L' in leg and 'F' in leg:  # FL
                self.phase_offsets[leg] = 0.5
            elif 'L' in leg and 'R' in leg and 'F' not in leg:  # RL
                self.phase_offsets[leg] = 0.5
        
        # Stance phase duration (50% of cycle for each side)
        self.stance_duration = 0.5
        
        # Blending zone for smooth transitions
        self.blend_zone = 0.1  # 10% of cycle for blending
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base velocity parameters
        self.vx_base = 0.4
        self.vx_modulation = 0.15

    def update_base_motion(self, phase, dt):
        """
        Update base with modulated forward velocity.
        Velocity peaks during stance phases when legs can provide propulsion.
        """
        # Smooth velocity modulation aligned with stance phases
        vx = self.vx_base + self.vx_modulation * np.abs(np.sin(2 * np.pi * phase))
        
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def smooth_blend(self, val1, val2, t):
        """Smooth cubic blending between two values."""
        # Cubic hermite blend with zero derivatives at endpoints
        blend = t * t * (3.0 - 2.0 * t)
        return val1 * (1.0 - blend) + val2 * blend

    def compute_windmill_swing_position(self, leg_name, swing_progress):
        """
        Compute windmill circular trajectory for swing phase.
        Uses reduced arc angle to limit joint extremes.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Use reduced arc angle: -pi/3 to +pi/3 (120 degrees instead of 180)
        # This limits the trajectory extrema while maintaining windmill character
        arc_start = -np.pi / 3.0
        arc_end = np.pi / 3.0
        angle = arc_start + (arc_end - arc_start) * swing_progress
        
        # Windmill circular motion with vertical offset
        foot[0] += self.windmill_radius_horizontal * np.sin(angle)
        foot[2] += self.vertical_offset + self.windmill_radius_vertical * (1.0 + np.cos(angle))
        
        return foot

    def compute_stance_position(self, leg_name, stance_progress):
        """
        Compute foot position during stance phase.
        Foot slides rearward in body frame as base moves forward.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Linear rearward motion during stance
        # Start slightly forward, end slightly rearward
        x_forward = self.windmill_radius_horizontal * 0.7
        x_rearward = -self.windmill_radius_horizontal * 0.7
        foot[0] += x_forward + (x_rearward - x_forward) * stance_progress
        
        # Keep foot on ground
        foot[2] = self.base_feet_pos_body[leg_name][2]
        
        return foot

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position using windmill circular trajectory with smooth blending.
        
        Windmill cycle:
        - Swing phase (0.0-0.5): Large vertical circular arc with blended transitions
        - Stance phase (0.5-1.0): Ground contact with rearward motion
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Define transition zones
        stance_to_swing_start = 0.5 - self.blend_zone / 2.0
        stance_to_swing_end = 0.5 + self.blend_zone / 2.0
        swing_to_stance_start = 1.0 - self.blend_zone / 2.0
        swing_to_stance_end = self.blend_zone / 2.0
        
        if leg_phase < stance_to_swing_start:
            # Pure stance phase
            stance_progress = leg_phase / 0.5
            foot = self.compute_stance_position(leg_name, stance_progress)
            
        elif leg_phase < stance_to_swing_end:
            # Blending from stance to swing
            blend_progress = (leg_phase - stance_to_swing_start) / self.blend_zone
            
            # Get positions from both phases
            stance_progress = stance_to_swing_start / 0.5
            foot_stance = self.compute_stance_position(leg_name, stance_progress)
            
            swing_progress = 0.0
            foot_swing = self.compute_windmill_swing_position(leg_name, swing_progress)
            
            # Blend between them
            foot = self.base_feet_pos_body[leg_name].copy()
            for i in range(3):
                foot[i] = self.smooth_blend(foot_stance[i], foot_swing[i], blend_progress)
            
        elif leg_phase < swing_to_stance_start:
            # Pure swing phase
            swing_progress = (leg_phase - 0.5) / 0.5
            foot = self.compute_windmill_swing_position(leg_name, swing_progress)
            
        else:
            # Blending from swing to stance (wraps around phase=1.0)
            if leg_phase >= swing_to_stance_start:
                blend_progress = (leg_phase - swing_to_stance_start) / self.blend_zone
            else:
                blend_progress = (leg_phase + (1.0 - swing_to_stance_start)) / self.blend_zone
            
            # Get positions from both phases
            swing_progress = 1.0
            foot_swing = self.compute_windmill_swing_position(leg_name, swing_progress)
            
            stance_progress = 0.0
            foot_stance = self.compute_stance_position(leg_name, stance_progress)
            
            # Blend between them
            foot = self.base_feet_pos_body[leg_name].copy()
            for i in range(3):
                foot[i] = self.smooth_blend(foot_swing[i], foot_stance[i], blend_progress)
        
        return foot