from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CHAMBERED_KICK_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Chambered kick advance gait for forward locomotion.
    
    Each leg sequentially chambers (retracts close to body with knee bent high)
    then explosively extends forward-downward, propelling the base forward.
    
    Sequence: RL → RR → FL → FR
    
    - Chamber phases: leg lifts and retracts close to body
    - Extension phases: leg drives forward-downward generating propulsion
    - Base surges forward during extension phases
    - Tripod/diagonal support maintained throughout
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - reduced to respect joint limits and ground contact
        self.chamber_height = 0.10  # Reduced from 0.15 to avoid extreme knee flexion
        self.chamber_retraction = 0.06  # Reduced from 0.10 to ease kinematic constraints
        self.chamber_lateral_retraction = 0.0  # Eliminated to simplify kinematics
        self.extension_forward = 0.12  # Forward reach during extension
        
        # Base velocity parameters
        self.v_prep = 0.3  # Low preparation velocity
        self.v_surge = 1.2  # High surge velocity during extension
        self.v_medium = 0.7  # Medium velocity for front leg phases
        self.v_settle = 0.2  # Settling velocity during reset
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        Phase structure:
        [0.0, 0.15]: RL chamber - low forward velocity
        [0.15, 0.3]: RL extend - high surge velocity
        [0.3, 0.45]: RR chamber - low forward velocity
        [0.45, 0.6]: RR extend - high surge velocity
        [0.6, 0.75]: FL chamber-extend - medium velocity
        [0.75, 0.9]: FR chamber-extend - medium velocity
        [0.9, 1.0]: neutral reset - settling velocity
        """
        
        if phase < 0.15:
            # RL chamber - preparation
            vx = self.v_prep
        elif phase < 0.3:
            # RL extend - surge
            vx = self.v_surge
        elif phase < 0.45:
            # RR chamber - preparation
            vx = self.v_prep
        elif phase < 0.6:
            # RR extend - surge
            vx = self.v_surge
        elif phase < 0.75:
            # FL chamber-extend - medium
            vx = self.v_medium
        elif phase < 0.9:
            # FR chamber-extend - medium
            vx = self.v_medium
        else:
            # Neutral reset - settling
            vx = self.v_settle
        
        self.vel_world = np.array([vx, 0.0, 0.0])
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
        Compute foot position in body frame for given leg and phase.
        
        Each leg follows chamber-extend-stance pattern in sequence:
        - RL: chamber [0.0-0.15], extend [0.15-0.3], stance [0.3-1.0]
        - RR: stance [0.0-0.3], chamber [0.3-0.45], extend [0.45-0.6], stance [0.6-1.0]
        - FL: stance [0.0-0.6], chamber [0.6-0.675], extend [0.675-0.75], stance [0.75-1.0]
        - FR: stance [0.0-0.75], chamber [0.75-0.825], extend [0.825-0.9], stance [0.9-1.0]
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg side for lateral offset
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            lateral_sign = 1.0  # Left side (positive y)
        else:
            lateral_sign = -1.0  # Right side (negative y)
        
        # RL: Rear Left
        if leg_name.startswith('RL'):
            if phase < 0.15:
                # Chamber phase: lift and retract
                progress = phase / 0.15
                # Smooth chamber motion with coordinated x-z trajectory
                chamber_factor = np.sin(0.5 * np.pi * progress)
                foot[0] += self.chamber_retraction * chamber_factor
                foot[1] -= lateral_sign * self.chamber_lateral_retraction * chamber_factor
                foot[2] += self.chamber_height * np.sin(np.pi * progress)
            elif phase < 0.3:
                # Extension phase: drive forward and return to ground
                progress = (phase - 0.15) / 0.15
                # Interpolate from chambered to extended position
                chamber_factor = np.cos(0.5 * np.pi * progress)
                extension_factor = np.sin(0.5 * np.pi * progress)
                foot[0] += self.chamber_retraction * chamber_factor
                foot[0] += self.extension_forward * extension_factor
                foot[1] -= lateral_sign * self.chamber_lateral_retraction * chamber_factor
                # Return to ground level smoothly
                foot[2] += self.chamber_height * np.sin(np.pi * (1.0 - progress))
                foot[2] = max(foot[2], 0.0)  # Enforce ground contact
            else:
                # Stance phase: maintain ground contact
                foot[2] = max(foot[2], 0.0)
        
        # RR: Rear Right
        elif leg_name.startswith('RR'):
            if phase < 0.3:
                # Stance phase
                foot[2] = max(foot[2], 0.0)
            elif phase < 0.45:
                # Chamber phase
                progress = (phase - 0.3) / 0.15
                chamber_factor = np.sin(0.5 * np.pi * progress)
                foot[0] += self.chamber_retraction * chamber_factor
                foot[1] -= lateral_sign * self.chamber_lateral_retraction * chamber_factor
                foot[2] += self.chamber_height * np.sin(np.pi * progress)
            elif phase < 0.6:
                # Extension phase
                progress = (phase - 0.45) / 0.15
                chamber_factor = np.cos(0.5 * np.pi * progress)
                extension_factor = np.sin(0.5 * np.pi * progress)
                foot[0] += self.chamber_retraction * chamber_factor
                foot[0] += self.extension_forward * extension_factor
                foot[1] -= lateral_sign * self.chamber_lateral_retraction * chamber_factor
                foot[2] += self.chamber_height * np.sin(np.pi * (1.0 - progress))
                foot[2] = max(foot[2], 0.0)
            else:
                # Stance phase
                foot[2] = max(foot[2], 0.0)
        
        # FL: Front Left
        elif leg_name.startswith('FL'):
            if phase < 0.6:
                # Stance phase
                foot[2] = max(foot[2], 0.0)
            elif phase < 0.675:
                # Chamber phase
                progress = (phase - 0.6) / 0.075
                chamber_factor = np.sin(0.5 * np.pi * progress)
                foot[0] += self.chamber_retraction * chamber_factor
                foot[1] -= lateral_sign * self.chamber_lateral_retraction * chamber_factor
                foot[2] += self.chamber_height * np.sin(np.pi * progress)
            elif phase < 0.75:
                # Extension phase
                progress = (phase - 0.675) / 0.075
                chamber_factor = np.cos(0.5 * np.pi * progress)
                extension_factor = np.sin(0.5 * np.pi * progress)
                foot[0] += self.chamber_retraction * chamber_factor
                foot[0] += self.extension_forward * extension_factor
                foot[1] -= lateral_sign * self.chamber_lateral_retraction * chamber_factor
                foot[2] += self.chamber_height * np.sin(np.pi * (1.0 - progress))
                foot[2] = max(foot[2], 0.0)
            else:
                # Stance phase
                foot[2] = max(foot[2], 0.0)
        
        # FR: Front Right
        elif leg_name.startswith('FR'):
            if phase < 0.75:
                # Stance phase
                foot[2] = max(foot[2], 0.0)
            elif phase < 0.825:
                # Chamber phase
                progress = (phase - 0.75) / 0.075
                chamber_factor = np.sin(0.5 * np.pi * progress)
                foot[0] += self.chamber_retraction * chamber_factor
                foot[1] -= lateral_sign * self.chamber_lateral_retraction * chamber_factor
                foot[2] += self.chamber_height * np.sin(np.pi * progress)
            elif phase < 0.9:
                # Extension phase
                progress = (phase - 0.825) / 0.075
                chamber_factor = np.cos(0.5 * np.pi * progress)
                extension_factor = np.sin(0.5 * np.pi * progress)
                foot[0] += self.chamber_retraction * chamber_factor
                foot[0] += self.extension_forward * extension_factor
                foot[1] -= lateral_sign * self.chamber_lateral_retraction * chamber_factor
                foot[2] += self.chamber_height * np.sin(np.pi * (1.0 - progress))
                foot[2] = max(foot[2], 0.0)
            else:
                # Stance phase
                foot[2] = max(foot[2], 0.0)
        
        return foot