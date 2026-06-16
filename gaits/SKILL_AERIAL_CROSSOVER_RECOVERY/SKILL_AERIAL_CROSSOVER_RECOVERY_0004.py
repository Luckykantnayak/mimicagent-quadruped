from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_AERIAL_CROSSOVER_RECOVERY_MotionGenerator(BaseMotionGenerator):
    """
    Aerial Crossover Recovery Skill.
    
    Motion cycle:
    - Phase [0.0, 0.25]: FL/RR aerial crossover (FR/RL grounded)
    - Phase [0.25, 0.4]: All four legs grounded, stabilization
    - Phase [0.4, 0.65]: FR/RL aerial crossover (FL/RR grounded)
    - Phase [0.65, 1.0]: All four legs grounded, recovery
    
    Base motion: sustained forward velocity with transient lateral and angular modulation
    during aerial crossover phases.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.vx_base = 0.8  # sustained forward velocity
        self.crossover_lateral_amplitude = 0.12  # lateral inward sweep during crossover
        self.crossover_vertical_clearance = 0.10  # vertical lift during aerial crossover
        self.skating_slide_length = 0.08  # aft slide distance during stance propulsion
        
        # Base angular modulation amplitudes
        self.vy_mod_amplitude = 0.15  # lateral velocity modulation during crossover
        self.roll_rate_amplitude = 0.4  # roll rate modulation during crossover
        self.yaw_rate_amplitude = 0.3  # yaw rate modulation during crossover
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        - Sustained forward velocity throughout
        - Lateral velocity and angular rates modulate during aerial crossover phases
        - Stabilize to zero modulation during landing/recovery phases
        """
        vx = self.vx_base
        vy = 0.0
        vz = 0.0
        
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase [0.0, 0.25]: FL/RR aerial crossover
        if 0.0 <= phase < 0.25:
            progress = phase / 0.25
            # Smooth modulation using sine envelope
            envelope = np.sin(np.pi * progress)
            vy = self.vy_mod_amplitude * envelope * np.sin(2 * np.pi * progress)
            roll_rate = self.roll_rate_amplitude * envelope * np.sin(2 * np.pi * progress)
            yaw_rate = self.yaw_rate_amplitude * envelope * np.sin(2 * np.pi * progress)
        
        # Phase [0.25, 0.4]: All legs grounded, stabilization
        elif 0.25 <= phase < 0.4:
            # Angular rates and lateral velocity return to zero
            pass
        
        # Phase [0.4, 0.65]: FR/RL aerial crossover
        elif 0.4 <= phase < 0.65:
            progress = (phase - 0.4) / 0.25
            # Smooth modulation using sine envelope, symmetric to first crossover
            envelope = np.sin(np.pi * progress)
            vy = -self.vy_mod_amplitude * envelope * np.sin(2 * np.pi * progress)
            roll_rate = -self.roll_rate_amplitude * envelope * np.sin(2 * np.pi * progress)
            yaw_rate = -self.yaw_rate_amplitude * envelope * np.sin(2 * np.pi * progress)
        
        # Phase [0.65, 1.0]: All legs grounded, recovery
        else:
            # Angular rates and lateral velocity return to zero
            pass
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase and leg assignment.
        
        Diagonal pairs alternate between:
        - Aerial crossover (swing with lateral inward motion and vertical clearance)
        - Grounded skating propulsion (stance with aft slide)
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this leg is in FL/RR group (group 1) or FR/RL group (group 2)
        is_group1 = leg_name.startswith('FL') or leg_name.startswith('RR')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        if is_group1:
            # Group 1: FL and RR
            # Aerial crossover during [0.0, 0.25]
            if 0.0 <= phase < 0.25:
                foot = self._compute_aerial_crossover(foot, phase / 0.25, is_left)
            # Landing during [0.25, 0.4]
            elif 0.25 <= phase < 0.4:
                foot = self._compute_landing_transition(foot, (phase - 0.25) / 0.15, is_left)
            # Grounded skating during [0.4, 0.65]
            elif 0.4 <= phase < 0.65:
                foot = self._compute_skating_stance(foot, (phase - 0.4) / 0.25)
            # Grounded recovery during [0.65, 1.0]
            else:
                foot = self._compute_skating_stance(foot, (phase - 0.65) / 0.35)
        else:
            # Group 2: FR and RL
            # Grounded skating during [0.0, 0.25]
            if 0.0 <= phase < 0.25:
                foot = self._compute_skating_stance(foot, phase / 0.25)
            # Grounded transition during [0.25, 0.4]
            elif 0.25 <= phase < 0.4:
                foot = self._compute_skating_stance(foot, (phase - 0.25) / 0.15)
            # Aerial crossover during [0.4, 0.65]
            elif 0.4 <= phase < 0.65:
                foot = self._compute_aerial_crossover(foot, (phase - 0.4) / 0.25, is_left)
            # Landing during [0.65, 1.0]
            else:
                foot = self._compute_landing_transition(foot, (phase - 0.65) / 0.35, is_left)
        
        return foot

    def _compute_aerial_crossover(self, foot_base, progress, is_left):
        """
        Compute aerial crossover trajectory with lateral inward sweep and vertical clearance.
        
        progress: [0, 1] within the crossover phase
        is_left: True if left leg (FL, RL), False if right leg (FR, RR)
        """
        foot = foot_base.copy()
        
        # Lateral crossover: sweep inward toward body centerline
        lateral_sign = 1.0 if is_left else -1.0
        lateral_offset = -lateral_sign * self.crossover_lateral_amplitude * np.sin(np.pi * progress)
        foot[1] += lateral_offset
        
        # Forward motion during crossover (small arc)
        foot[0] += 0.03 * np.sin(np.pi * progress)
        
        # Vertical clearance (arc shape)
        foot[2] += self.crossover_vertical_clearance * np.sin(np.pi * progress)
        
        return foot

    def _compute_landing_transition(self, foot_base, progress, is_left):
        """
        Smooth landing transition: foot returns to nominal position with damping.
        
        progress: [0, 1] within the landing phase
        """
        foot = foot_base.copy()
        
        # Small vertical damping at start of landing
        if progress < 0.3:
            foot[2] += 0.02 * (1.0 - progress / 0.3)
        
        # Small forward adjustment
        foot[0] += 0.01 * (1.0 - progress)
        
        return foot

    def _compute_skating_stance(self, foot_base, progress):
        """
        Skating propulsion stance: foot slides aft as base moves forward.
        
        progress: [0, 1] within the skating phase
        """
        foot = foot_base.copy()
        
        # Aft slide to simulate skating propulsion in body frame
        foot[0] -= self.skating_slide_length * progress
        
        return foot