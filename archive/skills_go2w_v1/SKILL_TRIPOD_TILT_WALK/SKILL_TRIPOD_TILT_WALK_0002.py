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
        
        # Gait timing parameters
        self.step_height = 0.24
        self.step_length = 0.12
        
        # Base foot positions in body frame with vertical offset for clearance
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            pos = v.copy()
            pos[2] += 0.025  # Add vertical clearance margin
            self.base_feet_pos_body[k] = pos
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters - reduced roll rate for stability
        self.vx_forward = 0.3
        self.roll_rate_magnitude = 0.9
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def smooth_step(self, x):
        """Smooth step function for blending transitions."""
        return 3 * x**2 - 2 * x**3

    def update_base_motion(self, phase, dt):
        """
        Update base pose with constant forward velocity and phase-dependent roll rate.
        """
        vx = self.vx_forward
        
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
            foot = self._compute_fl_trajectory(foot, phase)
        elif leg_name.startswith('FR'):
            foot = self._compute_fr_trajectory(foot, phase)
        elif leg_name.startswith('RL'):
            foot = self._compute_rl_trajectory(foot, phase)
        elif leg_name.startswith('RR'):
            foot = self._compute_rr_trajectory(foot, phase)
        
        return foot

    def _compute_fl_trajectory(self, foot_base, phase):
        """
        FL trajectory: stance [0.0-0.2], swing [0.2-0.8], stance [0.8-1.0]
        High swing during right tilt stance phase.
        """
        foot = foot_base.copy()
        
        # Blending parameters
        blend_window = 0.05
        
        if phase < 0.2:
            # Stance phase: retract foot backward with tilt compensation
            progress = phase / 0.2
            foot[0] -= self.step_length * (progress - 0.5)
            
            # Vertical compensation for left tilt (tilting toward this leg)
            # Slightly lower during left tilt stance
            tilt_compensation = -0.01 * np.sin(progress * np.pi)
            foot[2] += tilt_compensation
            
        elif phase < 0.2 + blend_window:
            # Smooth transition from stance to swing
            blend_progress = (phase - 0.2) / blend_window
            blend = self.smooth_step(blend_progress)
            
            # Stance contribution
            progress_stance = 1.0
            x_stance = foot_base[0] - self.step_length * (progress_stance - 0.5)
            z_stance = foot_base[2] - 0.01 * np.sin(progress_stance * np.pi)
            
            # Swing contribution
            swing_progress = 0.0
            x_swing = foot_base[0] + self.step_length * (swing_progress - 0.5)
            arc_progress = swing_progress * 2.0
            z_swing = foot_base[2] + self.step_height * arc_progress
            
            foot[0] = x_stance * (1 - blend) + x_swing * blend
            foot[2] = z_stance * (1 - blend) + z_swing * blend
            
        elif phase < 0.8 - blend_window:
            # Main swing phase: lift high and move forward
            swing_progress = (phase - 0.2) / 0.6
            
            # Forward motion during swing
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # High arc trajectory with smooth curve
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
            swing_progress = (0.8 - blend_window - 0.2) / 0.6
            x_swing = foot_base[0] + self.step_length * (swing_progress - 0.5)
            arc_progress = (1.0 - swing_progress) * 2.0
            z_swing = foot_base[2] + self.step_height * self.smooth_step(arc_progress)
            
            # Stance contribution
            progress_stance = 0.0
            x_stance = foot_base[0] - self.step_length * (progress_stance - 0.5) * 0.5
            z_stance = foot_base[2]
            
            foot[0] = x_swing * (1 - blend) + x_stance * blend
            foot[2] = z_swing * (1 - blend) + z_stance * blend
            
        else:
            # Stance re-establishment: foot settles
            progress = (phase - 0.8) / 0.2
            foot[0] -= self.step_length * (progress - 0.5) * 0.5
            
            # Vertical compensation for left tilt
            tilt_compensation = -0.01 * np.sin(progress * np.pi)
            foot[2] += tilt_compensation
        
        return foot

    def _compute_fr_trajectory(self, foot_base, phase):
        """
        FR trajectory: swing [0.0-0.4], stance [0.4-0.8], swing [0.8-1.0]
        High swing during left tilt stance phase.
        """
        foot = foot_base.copy()
        
        blend_window = 0.05
        
        if phase < 0.4 - blend_window:
            # Main swing phase: lift high and move forward
            swing_progress = phase / 0.4
            
            # Forward motion during swing
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # High arc trajectory with smooth curve
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
            swing_progress = (0.4 - blend_window) / 0.4
            x_swing = foot_base[0] + self.step_length * (swing_progress - 0.5)
            arc_progress = (1.0 - swing_progress) * 2.0
            z_swing = foot_base[2] + self.step_height * self.smooth_step(arc_progress)
            
            # Stance contribution
            progress_stance = 0.0
            x_stance = foot_base[0] - self.step_length * (progress_stance - 0.5)
            z_stance = foot_base[2]
            
            foot[0] = x_swing * (1 - blend) + x_stance * blend
            foot[2] = z_swing * (1 - blend) + z_stance * blend
            
        elif phase < 0.8:
            # Stance phase: retract foot backward with tilt compensation
            progress = (phase - 0.4) / 0.4
            foot[0] -= self.step_length * (progress - 0.5)
            
            # Vertical compensation for right tilt (tilting toward this leg)
            tilt_compensation = -0.01 * np.sin(progress * np.pi)
            foot[2] += tilt_compensation
            
        elif phase < 0.8 + blend_window:
            # Smooth transition from stance to swing
            blend_progress = (phase - 0.8) / blend_window
            blend = self.smooth_step(blend_progress)
            
            # Stance contribution
            progress_stance = 1.0
            x_stance = foot_base[0] - self.step_length * (progress_stance - 0.5)
            z_stance = foot_base[2] - 0.01 * np.sin(progress_stance * np.pi)
            
            # Swing contribution
            swing_progress = 0.0
            x_swing = foot_base[0] + self.step_length * (swing_progress - 0.5) * 0.5
            arc_progress = swing_progress * 2.0
            z_swing = foot_base[2] + self.step_height * arc_progress * 0.5
            
            foot[0] = x_stance * (1 - blend) + x_swing * blend
            foot[2] = z_stance * (1 - blend) + z_swing * blend
            
        else:
            # Swing initiation for next cycle
            swing_progress = (phase - 0.8) / 0.2
            foot[0] += self.step_length * (swing_progress - 0.5) * 0.5
            arc_progress = swing_progress * 2.0
            foot[2] += self.step_height * self.smooth_step(arc_progress) * 0.5
        
        return foot

    def _compute_rl_trajectory(self, foot_base, phase):
        """
        RL trajectory: continuous stance throughout entire cycle with vertical compensation.
        Provides constant rear-left support, lifts slightly during right tilt.
        """
        foot = foot_base.copy()
        
        # Forward-backward motion for body-frame positioning
        if phase < 0.5:
            progress = phase * 2.0
        else:
            progress = (phase - 0.5) * 2.0
        
        foot[0] -= self.step_length * 0.3 * (progress - 0.5)
        
        # Vertical compensation: lift during right tilt (phases 0.4-0.6)
        # Lower during left tilt (phases 0.0-0.2, 0.8-1.0)
        if phase < 0.2:
            # Left tilt - this leg is on tilted side, slight lower
            z_offset = -0.008 * np.sin((phase / 0.2) * np.pi)
        elif phase < 0.4:
            # Transition to right tilt - gradually lift
            transition_progress = (phase - 0.2) / 0.2
            z_offset = 0.015 * self.smooth_step(transition_progress)
        elif phase < 0.6:
            # Right tilt - this leg is on raised side, lift to clear
            z_offset = 0.015 + 0.01 * np.sin(((phase - 0.4) / 0.2) * np.pi)
        elif phase < 0.8:
            # Transition to left tilt - gradually lower
            transition_progress = (phase - 0.6) / 0.2
            z_offset = 0.015 * (1 - self.smooth_step(transition_progress))
        else:
            # Left tilt re-establishment
            z_offset = -0.008 * np.sin(((phase - 0.8) / 0.2) * np.pi)
        
        foot[2] += z_offset
        
        return foot

    def _compute_rr_trajectory(self, foot_base, phase):
        """
        RR trajectory: continuous stance throughout entire cycle with vertical compensation.
        Provides constant rear-right support, lifts slightly during left tilt.
        """
        foot = foot_base.copy()
        
        # Forward-backward motion for body-frame positioning
        if phase < 0.5:
            progress = phase * 2.0
        else:
            progress = (phase - 0.5) * 2.0
        
        foot[0] -= self.step_length * 0.3 * (progress - 0.5)
        
        # Vertical compensation: lift during left tilt (phases 0.0-0.2, 0.8-1.0)
        # Lower during right tilt (phases 0.4-0.6)
        if phase < 0.2:
            # Left tilt - this leg is on raised side, lift to clear
            z_offset = 0.015 + 0.01 * np.sin((phase / 0.2) * np.pi)
        elif phase < 0.4:
            # Transition to right tilt - gradually lower
            transition_progress = (phase - 0.2) / 0.2
            z_offset = 0.015 * (1 - self.smooth_step(transition_progress))
        elif phase < 0.6:
            # Right tilt - this leg is on tilted side, slight lower
            z_offset = -0.008 * np.sin(((phase - 0.4) / 0.2) * np.pi)
        elif phase < 0.8:
            # Transition to left tilt - gradually lift
            transition_progress = (phase - 0.6) / 0.2
            z_offset = 0.015 * self.smooth_step(transition_progress)
        else:
            # Left tilt re-establishment - this leg is on raised side
            z_offset = 0.015 + 0.01 * np.sin(((phase - 0.8) / 0.2) * np.pi)
        
        foot[2] += z_offset
        
        return foot