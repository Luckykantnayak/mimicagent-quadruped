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
        
        # Diagonal trot parameters
        self.duty = 0.6
        self.step_length = 0.06
        self.step_height = 0.06
        
        # Base foot positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Calculate nominal standing height from base foot positions
        avg_foot_z = np.mean([pos[2] for pos in initial_foot_positions_body.values()])
        self.standing_height = abs(avg_foot_z)
        
        # Diagonal trot phase offsets: FL+RR vs FR+RL
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0
            else:  # FR or RL
                self.phase_offsets[leg] = 0.5
        
        # Orbital motion parameters
        self.vx = 0.3  # forward velocity (m/s)
        self.yaw_rate = 0.5  # constant yaw rate for circular orbit (rad/s)
        
        # Roll oscillation parameters
        self.roll_rate_amp = 1.2  # amplitude of roll rate oscillation (rad/s)
        self.roll_freq = self.freq  # roll oscillates once per cycle
        
        # Lateral extension gain: reduced to prevent joint limit violations
        self.lateral_gain = 0.07  # m/rad
        
        # Blending window for smooth stance-swing transitions
        self.blend_window = 0.05  # 5% of cycle
        
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
        
        Roll rate oscillates: negative in [0, 0.25] and [0.5, 0.75],
                              positive in [0.25, 0.5] and [0.75, 1.0]
        This creates left-right-left-right roll pattern synchronized with orbit.
        """
        # Constant forward velocity for orbit
        vx = self.vx
        
        # Sinusoidal roll rate: -sin(2*pi*phase) for desired pattern
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
        
        # Maintain standing height (z may drift slightly due to numerical integration)
        self.root_pos[2] = self.standing_height
        
        # Update current roll angle for leg lateral modulation
        roll, pitch, yaw = quat_to_euler(self.root_quat)
        self.current_roll = roll

    def smooth_step(self, x):
        """Smoothstep function for C1 continuous blending: 3x^2 - 2x^3"""
        x = np.clip(x, 0.0, 1.0)
        return 3.0 * x * x - 2.0 * x * x * x

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with:
        1. Diagonal trot gait pattern (forward/vertical motion) with smooth transitions
        2. Lateral modulation based on current roll angle
        
        Outer legs (opposite to roll direction) extend outward.
        Inner legs (same side as roll) retract inward.
        """
        # Get leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is on left or right side
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compute lateral modulation based on roll angle
        if is_left_leg:
            lateral_offset = self.lateral_gain * self.current_roll
        else:
            lateral_offset = -self.lateral_gain * self.current_roll
        
        foot[1] += lateral_offset
        
        # Compute stance and swing profiles with smooth blending
        if leg_phase < self.duty - self.blend_window:
            # Pure stance phase
            progress = leg_phase / self.duty
            foot[0] -= self.step_length * (progress - 0.5)
            
        elif leg_phase < self.duty + self.blend_window:
            # Blend region around stance-to-swing transition
            blend_progress = (leg_phase - (self.duty - self.blend_window)) / (2.0 * self.blend_window)
            blend_factor = self.smooth_step(blend_progress)
            
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
            blend_factor = self.smooth_step(blend_progress)
            
            # Swing contribution
            swing_progress = (leg_phase - self.duty) / (1.0 - self.duty)
            swing_x = self.step_length * (swing_progress - 0.5)
            swing_angle = np.pi * swing_progress
            swing_z = self.step_height * np.sin(swing_angle)
            
            # Next stance contribution (wraparound)
            next_phase = (leg_phase - 1.0) if leg_phase >= 1.0 else leg_phase
            stance_progress = next_phase / self.duty
            stance_x = -self.step_length * (stance_progress - 0.5)
            
            # Blend between swing and stance
            foot[0] += swing_x * (1.0 - blend_factor) + stance_x * blend_factor
            foot[2] += swing_z * (1.0 - blend_factor)
        
        return foot