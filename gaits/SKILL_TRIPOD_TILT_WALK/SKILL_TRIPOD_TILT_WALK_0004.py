from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_TRIPOD_TILT_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Tripod tilt walk with alternating tripod support and exaggerated lateral roll tilting.
    
    Motion structure:
    - Phase [0.0, 0.2]: Left tripod stance (FL, RR, RL), FR swings high, heavy left tilt
    - Phase [0.2, 0.4]: Transition to right tilt, FL/FR swap roles
    - Phase [0.4, 0.6]: Right tripod stance (FR, RL, RR), FL swings high, heavy right tilt
    - Phase [0.6, 0.8]: Transition to left tilt, FR/FL swap roles
    - Phase [0.8, 1.0]: Re-establish left tripod stance
    
    Base motion:
    - Constant forward velocity
    - Roll rate alternates sign to create rocking motion
    
    Leg motion:
    - FL and FR alternate between stance and high swing
    - RL and RR remain in continuous contact (rear support)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Gait timing parameters - reduced for workspace compatibility
        self.step_height = 0.14
        self.step_length = 0.09
        
        # Base foot positions in body frame with minimal vertical offset
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            pos = v.copy()
            # Minimal clearance to avoid accumulation with swing height
            if k.startswith('FL') or k.startswith('FR'):
                pos[2] += 0.02
            else:
                pos[2] += 0.015
            self.base_feet_pos_body[k] = pos
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters
        self.vx_forward = 0.3
        self.roll_rate_magnitude = 0.9
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Track accumulated roll for dynamic compensation
        self.accumulated_roll = 0.0

    def smooth_step(self, x):
        """Smooth step function for blending transitions."""
        x = np.clip(x, 0.0, 1.0)
        return 3 * x**2 - 2 * x**3

    def get_dynamic_tilt_compensation(self, leg_name, phase):
        """
        Compute vertical compensation based on body roll and lateral foot position.
        Prevents ground penetration during tilted body configurations.
        """
        # Get lateral offset of foot from body centerline
        lateral_offset = abs(self.base_feet_pos_body[leg_name][1])
        
        # Estimate roll angle based on phase and roll rate
        if phase < 0.2:
            roll_estimate = -0.15  # Left tilt
        elif phase < 0.4:
            # Transition from left to right
            transition = (phase - 0.2) / 0.2
            roll_estimate = -0.15 + 0.30 * self.smooth_step(transition)
        elif phase < 0.6:
            roll_estimate = 0.15  # Right tilt
        elif phase < 0.8:
            # Transition from right to left
            transition = (phase - 0.6) / 0.2
            roll_estimate = 0.15 - 0.30 * self.smooth_step(transition)
        else:
            roll_estimate = -0.15  # Left tilt
        
        # Geometric vertical displacement due to roll
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            # Left legs - go down during right tilt (positive roll)
            z_offset = -roll_estimate * lateral_offset
        else:
            # Right legs - go down during left tilt (negative roll)
            z_offset = roll_estimate * lateral_offset
        
        return z_offset

    def update_base_motion(self, phase, dt):
        """
        Update base pose with constant forward velocity and phase-dependent roll rate.
        """
        vx = self.vx_forward
        
        # Roll rate schedule aligned with tripod phases
        if phase < 0.2:
            roll_rate = -self.roll_rate_magnitude
        elif phase < 0.4:
            roll_rate = self.roll_rate_magnitude
        elif phase < 0.6:
            roll_rate = self.roll_rate_magnitude
        elif phase < 0.8:
            roll_rate = -self.roll_rate_magnitude
        else:
            roll_rate = -self.roll_rate_magnitude
        
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase and leg role.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_name.startswith('FL'):
            foot = self._compute_fl_trajectory(foot, phase, leg_name)
        elif leg_name.startswith('FR'):
            foot = self._compute_fr_trajectory(foot, phase, leg_name)
        elif leg_name.startswith('RL'):
            foot = self._compute_rl_trajectory(foot, phase, leg_name)
        elif leg_name.startswith('RR'):
            foot = self._compute_rr_trajectory(foot, phase, leg_name)
        
        return foot

    def _compute_fl_trajectory(self, foot_base, phase, leg_name):
        """
        FL trajectory: stance [0.0-0.2], swing [0.2-0.8], stance [0.8-1.0]
        High swing during right tilt stance phase with conservative reach.
        """
        foot = foot_base.copy()
        
        blend_window = 0.04
        
        if phase < 0.2:
            # Stance phase
            progress = phase / 0.2
            foot[0] -= self.step_length * 0.4 * (2 * progress - 1)
            
            # Dynamic tilt compensation
            z_comp = self.get_dynamic_tilt_compensation(leg_name, phase)
            foot[2] += z_comp
            
        elif phase < 0.2 + blend_window:
            # Smooth transition from stance to swing
            blend_progress = (phase - 0.2) / blend_window
            blend = self.smooth_step(blend_progress)
            
            # Stance contribution
            x_stance = foot_base[0] - self.step_length * 0.4
            z_stance = foot_base[2] + self.get_dynamic_tilt_compensation(leg_name, 0.2)
            
            # Swing contribution - initial lift with minimal forward motion
            x_swing = foot_base[0] - self.step_length * 0.35
            z_swing = foot_base[2] + self.step_height * 0.15
            
            foot[0] = x_stance * (1 - blend) + x_swing * blend
            foot[2] = z_stance * (1 - blend) + z_swing * blend
            
        elif phase < 0.8 - blend_window:
            # Main swing phase: conservative horizontal reach at peak height
            swing_progress = (phase - (0.2 + blend_window)) / (0.6 - 2 * blend_window)
            
            # Forward motion during swing - strongly reduced at peak
            if swing_progress < 0.3:
                x_factor = -0.4 + swing_progress * 0.5
            elif swing_progress < 0.7:
                # Near peak - minimal forward displacement
                x_factor = -0.25
            else:
                x_factor = -0.25 + (swing_progress - 0.7) * 1.5
            foot[0] += self.step_length * x_factor
            
            # Smooth arc trajectory with reduced step height
            if swing_progress < 0.5:
                arc_progress = swing_progress * 2.0
            else:
                arc_progress = (1.0 - swing_progress) * 2.0
            foot[2] += self.step_height * self.smooth_step(arc_progress)
            
        elif phase < 0.8:
            # Smooth transition from swing to stance
            blend_progress = (phase - (0.8 - blend_window)) / blend_window
            blend = self.smooth_step(blend_progress)
            
            # Swing contribution
            x_swing = foot_base[0] + self.step_length * 0.2
            z_swing = foot_base[2] + self.step_height * 0.15
            
            # Stance contribution
            x_stance = foot_base[0] + self.step_length * 0.4
            z_stance = foot_base[2] + self.get_dynamic_tilt_compensation(leg_name, 0.8)
            
            foot[0] = x_swing * (1 - blend) + x_stance * blend
            foot[2] = z_swing * (1 - blend) + z_stance * blend
            
        else:
            # Stance re-establishment
            progress = (phase - 0.8) / 0.2
            foot[0] += self.step_length * 0.4 * (1 - 2 * progress)
            
            # Dynamic tilt compensation
            z_comp = self.get_dynamic_tilt_compensation(leg_name, phase)
            foot[2] += z_comp
        
        return foot

    def _compute_fr_trajectory(self, foot_base, phase, leg_name):
        """
        FR trajectory: swing [0.0-0.4], stance [0.4-0.8], swing start [0.8-1.0]
        High swing during left tilt stance phase with conservative reach.
        """
        foot = foot_base.copy()
        
        blend_window = 0.04
        
        if phase < 0.4 - blend_window:
            # Main swing phase with conservative reach at peak height
            swing_progress = phase / (0.4 - blend_window)
            
            # Forward motion during swing - strongly reduced at peak
            if swing_progress < 0.3:
                x_factor = -0.4 + swing_progress * 0.5
            elif swing_progress < 0.7:
                # Near peak - minimal forward displacement
                x_factor = -0.25
            else:
                x_factor = -0.25 + (swing_progress - 0.7) * 1.5
            foot[0] += self.step_length * x_factor
            
            # Full height arc trajectory
            if swing_progress < 0.5:
                arc_progress = swing_progress * 2.0
            else:
                arc_progress = (1.0 - swing_progress) * 2.0
            foot[2] += self.step_height * self.smooth_step(arc_progress)
            
        elif phase < 0.4:
            # Smooth transition from swing to stance
            blend_progress = (phase - (0.4 - blend_window)) / blend_window
            blend = self.smooth_step(blend_progress)
            
            # Swing contribution
            x_swing = foot_base[0] + self.step_length * 0.2
            z_swing = foot_base[2] + self.step_height * 0.15
            
            # Stance contribution
            x_stance = foot_base[0] + self.step_length * 0.4
            z_stance = foot_base[2] + self.get_dynamic_tilt_compensation(leg_name, 0.4)
            
            foot[0] = x_swing * (1 - blend) + x_stance * blend
            foot[2] = z_swing * (1 - blend) + z_stance * blend
            
        elif phase < 0.8:
            # Stance phase with dynamic compensation
            progress = (phase - 0.4) / 0.4
            foot[0] += self.step_length * 0.4 * (1 - 2 * progress)
            
            # Dynamic tilt compensation
            z_comp = self.get_dynamic_tilt_compensation(leg_name, phase)
            foot[2] += z_comp
            
        elif phase < 0.8 + blend_window:
            # Smooth transition from stance to swing
            blend_progress = (phase - 0.8) / blend_window
            blend = self.smooth_step(blend_progress)
            
            # Stance contribution
            x_stance = foot_base[0] - self.step_length * 0.4
            z_stance = foot_base[2] + self.get_dynamic_tilt_compensation(leg_name, 0.8)
            
            # Swing contribution - initial lift with minimal forward motion
            x_swing = foot_base[0] - self.step_length * 0.35
            z_swing = foot_base[2] + self.step_height * 0.2
            
            foot[0] = x_stance * (1 - blend) + x_swing * blend
            foot[2] = z_stance * (1 - blend) + z_swing * blend
            
        else:
            # Swing initiation for next cycle
            swing_progress = (phase - (0.8 + blend_window)) / (0.2 - blend_window)
            
            # Conservative forward motion during early swing
            x_factor = -0.4 + swing_progress * 0.15
            foot[0] += self.step_length * x_factor
            
            # Gradual height increase
            arc_progress = swing_progress
            foot[2] += self.step_height * self.smooth_step(arc_progress)
        
        return foot

    def _compute_rl_trajectory(self, foot_base, phase, leg_name):
        """
        RL trajectory: continuous stance throughout entire cycle with vertical compensation.
        """
        foot = foot_base.copy()
        
        # Smooth forward-backward motion with reduced amplitude
        if phase < 0.5:
            progress = self.smooth_step(phase * 2.0)
        else:
            progress = self.smooth_step((phase - 0.5) * 2.0)
        
        foot[0] -= self.step_length * 0.2 * np.sin(progress * np.pi)
        
        # Dynamic tilt compensation
        z_comp = self.get_dynamic_tilt_compensation(leg_name, phase)
        foot[2] += z_comp
        
        # Additional phase-specific compensation for stability
        if 0.4 <= phase < 0.6:
            # Right tilt peak - left leg needs extra clearance
            foot[2] += 0.01 * np.sin(((phase - 0.4) / 0.2) * np.pi)
        
        return foot

    def _compute_rr_trajectory(self, foot_base, phase, leg_name):
        """
        RR trajectory: continuous stance throughout entire cycle with vertical compensation.
        """
        foot = foot_base.copy()
        
        # Smooth forward-backward motion with reduced amplitude
        if phase < 0.5:
            progress = self.smooth_step(phase * 2.0)
        else:
            progress = self.smooth_step((phase - 0.5) * 2.0)
        
        foot[0] -= self.step_length * 0.2 * np.sin(progress * np.pi)
        
        # Dynamic tilt compensation
        z_comp = self.get_dynamic_tilt_compensation(leg_name, phase)
        foot[2] += z_comp
        
        # Additional phase-specific compensation for stability
        if phase < 0.2 or phase >= 0.8:
            # Left tilt phases - right leg needs extra clearance
            if phase < 0.2:
                foot[2] += 0.01 * np.sin((phase / 0.2) * np.pi)
            else:
                foot[2] += 0.01 * np.sin(((phase - 0.8) / 0.2) * np.pi)
        
        return foot