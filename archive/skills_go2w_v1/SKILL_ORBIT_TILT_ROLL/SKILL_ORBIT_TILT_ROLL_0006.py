from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ORBIT_TILT_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Orbit with synchronized roll oscillation.
    
    The robot executes a continuous circular orbit while rhythmically rolling 
    its base left and right. Legs dynamically adjust their lateral stance width:
    outer legs extend, inner legs retract, to support the tilted base throughout 
    the circular path.
    
    - Base motion: constant forward velocity + constant yaw rate (orbit)
                   + sinusoidal roll rate (oscillating tilt)
    - Leg motion: diagonal trot gait with lateral modulation based on roll angle
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Diagonal trot parameters - adjusted for continuous ground contact
        self.duty = 0.68
        self.step_length = 0.06
        self.step_height = 0.048
        
        # Base foot positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Calculate nominal standing height from base foot positions with added clearance
        avg_foot_z = np.mean([pos[2] for pos in initial_foot_positions_body.values()])
        base_standing_height = abs(avg_foot_z)
        
        # Add clearance to account for geometric height loss during roll with lateral extension
        clearance_margin = 0.04
        self.standing_height = base_standing_height + clearance_margin
        
        # Diagonal trot phase offsets: FL+RR vs FR+RL - adjusted to prevent flight
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0
            else:  # FR or RL
                self.phase_offsets[leg] = 0.57
        
        # Orbital motion parameters
        self.vx = 0.3
        self.yaw_rate = 0.5
        
        # Roll oscillation parameters
        self.roll_rate_amp = 0.9
        self.roll_freq = self.freq
        
        # Lateral extension gain - reduced to prevent joint limit violations
        self.lateral_gain = 0.022
        
        # Roll magnitude threshold for adaptive attenuation
        self.roll_threshold = 0.3
        
        # Blending window for smooth stance-swing transitions - reduced to minimize overlap
        self.blend_window = 0.07
        
        # Base state with proper standing height
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.standing_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track integrated roll angle for leg modulation
        self.current_roll = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward velocity, constant yaw rate,
        and sinusoidal roll rate. Maintains standing height.
        """
        # Constant forward velocity for orbit
        vx = self.vx
        
        # Sinusoidal roll rate
        roll_rate = -self.roll_rate_amp * np.sin(2 * np.pi * phase)
        
        # Constant yaw rate for circular orbit
        yaw_rate = self.yaw_rate
        
        # Set velocity commands
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Maintain standing height
        self.root_pos[2] = self.standing_height
        
        # Update current roll angle for leg lateral modulation
        roll, pitch, yaw = quat_to_euler(self.root_quat)
        self.current_roll = roll

    def smooth_step_quintic(self, x):
        """Quintic smoothstep function for C2 continuous blending"""
        x = np.clip(x, 0.0, 1.0)
        return 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3

    def compute_lateral_gain_scaling(self, leg_phase):
        """
        Compute adaptive lateral gain scaling based on leg phase and roll magnitude.
        Reduces lateral modulation during swing to preserve workspace margin.
        """
        # Determine if leg is in swing phase
        is_swing = leg_phase >= self.duty
        
        # Base scaling: more aggressive reduction during swing
        if is_swing:
            phase_scale = 0.45
        else:
            phase_scale = 1.0
        
        # Roll-magnitude-dependent attenuation: soft saturation at high roll angles
        abs_roll = abs(self.current_roll)
        if abs_roll > self.roll_threshold:
            roll_scale = self.roll_threshold / abs_roll
            roll_scale = np.clip(roll_scale, 0.5, 1.0)
        else:
            roll_scale = 1.0
        
        return phase_scale * roll_scale

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with:
        1. Diagonal trot gait pattern with smooth transitions
        2. Adaptive lateral modulation based on current roll angle and leg phase
        """
        # Get leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is on left or right side
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compute adaptive lateral gain scaling
        lateral_scale = self.compute_lateral_gain_scaling(leg_phase)
        
        # Apply lateral modulation based on roll angle with adaptive scaling
        if is_left_leg:
            lateral_offset = self.lateral_gain * self.current_roll * lateral_scale
        else:
            lateral_offset = -self.lateral_gain * self.current_roll * lateral_scale
        
        foot[1] += lateral_offset
        
        # Compute stance and swing profiles with smooth blending
        if leg_phase < self.duty - self.blend_window:
            # Pure stance phase
            progress = leg_phase / self.duty
            foot[0] -= self.step_length * (progress - 0.5)
            
        elif leg_phase < self.duty + self.blend_window:
            # Blend region around stance-to-swing transition
            blend_progress = (leg_phase - (self.duty - self.blend_window)) / (2.0 * self.blend_window)
            blend_factor = self.smooth_step_quintic(blend_progress)
            
            # Stance contribution
            stance_progress = leg_phase / self.duty
            stance_x = -self.step_length * (stance_progress - 0.5)
            
            # Swing contribution
            swing_phase_start = self.duty
            swing_progress = (leg_phase - swing_phase_start) / (1.0 - self.duty)
            swing_progress = np.clip(swing_progress, 0.0, 1.0)
            swing_x = self.step_length * (swing_progress - 0.5)
            swing_angle = np.pi * swing_progress
            swing_z = self.step_height * np.sin(swing_angle)
            
            # Blend between stance and swing
            foot[0] += stance_x * (1.0 - blend_factor) + swing_x * blend_factor
            foot[2] += swing_z * blend_factor
            
        elif leg_phase < 1.0 - self.blend_window:
            # Pure swing phase
            swing_progress = (leg_phase - self.duty) / (1.0 - self.duty)
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            swing_angle = np.pi * swing_progress
            foot[2] += self.step_height * np.sin(swing_angle)
            
        else:
            # Blend region around swing-to-stance transition
            blend_progress = (leg_phase - (1.0 - self.blend_window)) / (2.0 * self.blend_window)
            blend_factor = self.smooth_step_quintic(blend_progress)
            
            # Swing contribution
            swing_progress = (leg_phase - self.duty) / (1.0 - self.duty)
            swing_x = self.step_length * (swing_progress - 0.5)
            swing_angle = np.pi * swing_progress
            swing_z = self.step_height * np.sin(swing_angle)
            
            # Next stance contribution
            next_phase = leg_phase - 1.0
            stance_progress = next_phase / self.duty
            stance_x = -self.step_length * (stance_progress - 0.5)
            
            # Blend between swing and stance
            foot[0] += swing_x * (1.0 - blend_factor) + stance_x * blend_factor
            foot[2] += swing_z * (1.0 - blend_factor)
        
        return foot