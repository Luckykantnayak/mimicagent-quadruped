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
        self.step_height = 0.18
        self.step_length = 0.12
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters
        self.vx_forward = 0.3
        self.roll_rate_magnitude = 1.2
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base pose with constant forward velocity and phase-dependent roll rate.
        
        Roll rate pattern:
        - [0.0, 0.2]: negative (tilting left)
        - [0.2, 0.4]: positive (tilting right)
        - [0.4, 0.6]: positive (maintaining right tilt)
        - [0.6, 0.8]: negative (tilting left)
        - [0.8, 1.0]: negative (maintaining left tilt)
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
        
        FL: stance [0.0-0.2], swing [0.2-0.8], stance [0.8-1.0]
        FR: swing [0.0-0.4], stance [0.4-0.8], swing [0.8-1.0]
        RL: continuous stance (never swings)
        RR: continuous stance (never swings)
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
        
        if phase < 0.2:
            # Stance phase: retract foot backward
            progress = phase / 0.2
            foot[0] -= self.step_length * (progress - 0.5)
        elif phase < 0.8:
            # Swing phase: lift high and move forward
            swing_progress = (phase - 0.2) / 0.6
            
            # Forward motion during swing
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # High arc trajectory
            if swing_progress < 0.5:
                arc_progress = swing_progress * 2.0
            else:
                arc_progress = (1.0 - swing_progress) * 2.0
            foot[2] += self.step_height * arc_progress
        else:
            # Stance re-establishment: foot settles
            progress = (phase - 0.8) / 0.2
            foot[0] -= self.step_length * (progress - 0.5) * 0.5
        
        return foot

    def _compute_fr_trajectory(self, foot_base, phase):
        """
        FR trajectory: swing [0.0-0.4], stance [0.4-0.8], swing [0.8-1.0]
        High swing during left tilt stance phase.
        """
        foot = foot_base.copy()
        
        if phase < 0.4:
            # Swing phase: lift high and move forward
            swing_progress = phase / 0.4
            
            # Forward motion during swing
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # High arc trajectory
            if swing_progress < 0.5:
                arc_progress = swing_progress * 2.0
            else:
                arc_progress = (1.0 - swing_progress) * 2.0
            foot[2] += self.step_height * arc_progress
        elif phase < 0.8:
            # Stance phase: retract foot backward
            progress = (phase - 0.4) / 0.4
            foot[0] -= self.step_length * (progress - 0.5)
        else:
            # Swing initiation for next cycle
            swing_progress = (phase - 0.8) / 0.2
            foot[0] += self.step_length * (swing_progress - 0.5) * 0.5
            arc_progress = swing_progress * 2.0
            foot[2] += self.step_height * arc_progress * 0.5
        
        return foot

    def _compute_rl_trajectory(self, foot_base, phase):
        """
        RL trajectory: continuous stance throughout entire cycle.
        Provides constant rear-left support.
        """
        foot = foot_base.copy()
        
        # Minimal forward-backward motion to maintain body-frame positioning
        # during base translation
        if phase < 0.5:
            progress = phase * 2.0
        else:
            progress = (phase - 0.5) * 2.0
        
        foot[0] -= self.step_length * 0.3 * (progress - 0.5)
        
        return foot

    def _compute_rr_trajectory(self, foot_base, phase):
        """
        RR trajectory: continuous stance throughout entire cycle.
        Provides constant rear-right support.
        """
        foot = foot_base.copy()
        
        # Minimal forward-backward motion to maintain body-frame positioning
        # during base translation
        if phase < 0.5:
            progress = phase * 2.0
        else:
            progress = (phase - 0.5) * 2.0
        
        foot[0] -= self.step_length * 0.3 * (progress - 0.5)
        
        return foot