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
        
        # Motion parameters - reduced to respect joint limits
        self.chamber_height_rear = 0.040  # Reduced vertical lift for rear legs
        self.chamber_height_front = 0.028  # Further reduced for front legs
        self.chamber_retraction_rear = 0.070  # Substantial rearward retraction for rear legs
        self.chamber_retraction_front = 0.045  # Moderate retraction for front legs
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

    def smooth_transition(self, t):
        """Smooth interpolation function using cosine for C1 continuity"""
        return 0.5 * (1.0 - np.cos(np.pi * t))

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
        
        # Determine leg-specific parameters
        is_rear_leg = leg_name.startswith('RL') or leg_name.startswith('RR')
        chamber_height = self.chamber_height_rear if is_rear_leg else self.chamber_height_front
        chamber_retraction = self.chamber_retraction_rear if is_rear_leg else self.chamber_retraction_front
        
        # RL: Rear Left
        if leg_name.startswith('RL'):
            if phase < 0.15:
                # Chamber phase: retraction leads, then vertical lift follows
                progress = phase / 0.15
                
                # Horizontal retraction starts early and reaches max by mid-phase
                retraction_progress = min(progress * 1.5, 1.0)
                retraction_amount = chamber_retraction * self.smooth_transition(retraction_progress)
                foot[0] -= retraction_amount  # Move rearward
                
                # Vertical lift accelerates in second half
                if progress > 0.3:
                    lift_progress = (progress - 0.3) / 0.7
                    lift_amount = chamber_height * self.smooth_transition(lift_progress)
                    foot[2] += lift_amount
                
            elif phase < 0.3:
                # Extension phase: descend while driving forward
                progress = (phase - 0.15) / 0.15
                smooth_prog = self.smooth_transition(progress)
                
                # Initial chambered position
                retraction_decay = 1.0 - smooth_prog
                foot[0] -= chamber_retraction * retraction_decay
                
                # Forward extension builds throughout phase
                foot[0] += self.extension_forward * smooth_prog
                
                # Vertical descent completes by mid-phase
                if progress < 0.6:
                    descent_progress = progress / 0.6
                    remaining_height = chamber_height * (1.0 - self.smooth_transition(descent_progress))
                    foot[2] += remaining_height
                
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
                
                retraction_progress = min(progress * 1.5, 1.0)
                retraction_amount = chamber_retraction * self.smooth_transition(retraction_progress)
                foot[0] -= retraction_amount
                
                if progress > 0.3:
                    lift_progress = (progress - 0.3) / 0.7
                    lift_amount = chamber_height * self.smooth_transition(lift_progress)
                    foot[2] += lift_amount
                
            elif phase < 0.6:
                # Extension phase
                progress = (phase - 0.45) / 0.15
                smooth_prog = self.smooth_transition(progress)
                
                retraction_decay = 1.0 - smooth_prog
                foot[0] -= chamber_retraction * retraction_decay
                foot[0] += self.extension_forward * smooth_prog
                
                if progress < 0.6:
                    descent_progress = progress / 0.6
                    remaining_height = chamber_height * (1.0 - self.smooth_transition(descent_progress))
                    foot[2] += remaining_height
                
            else:
                # Stance phase
                foot[2] = max(foot[2], 0.0)
        
        # FL: Front Left
        elif leg_name.startswith('FL'):
            if phase < 0.6:
                # Stance phase
                foot[2] = max(foot[2], 0.0)
            elif phase < 0.675:
                # Chamber phase (shorter duration)
                progress = (phase - 0.6) / 0.075
                
                retraction_progress = min(progress * 1.5, 1.0)
                retraction_amount = chamber_retraction * self.smooth_transition(retraction_progress)
                foot[0] -= retraction_amount
                
                if progress > 0.3:
                    lift_progress = (progress - 0.3) / 0.7
                    lift_amount = chamber_height * self.smooth_transition(lift_progress)
                    foot[2] += lift_amount
                
            elif phase < 0.75:
                # Extension phase (shorter duration)
                progress = (phase - 0.675) / 0.075
                smooth_prog = self.smooth_transition(progress)
                
                retraction_decay = 1.0 - smooth_prog
                foot[0] -= chamber_retraction * retraction_decay
                foot[0] += self.extension_forward * 0.75 * smooth_prog  # Reduced extension for front
                
                if progress < 0.6:
                    descent_progress = progress / 0.6
                    remaining_height = chamber_height * (1.0 - self.smooth_transition(descent_progress))
                    foot[2] += remaining_height
                
            else:
                # Stance phase
                foot[2] = max(foot[2], 0.0)
        
        # FR: Front Right
        elif leg_name.startswith('FR'):
            if phase < 0.75:
                # Stance phase
                foot[2] = max(foot[2], 0.0)
            elif phase < 0.825:
                # Chamber phase (shorter duration)
                progress = (phase - 0.75) / 0.075
                
                retraction_progress = min(progress * 1.5, 1.0)
                retraction_amount = chamber_retraction * self.smooth_transition(retraction_progress)
                foot[0] -= retraction_amount
                
                if progress > 0.3:
                    lift_progress = (progress - 0.3) / 0.7
                    lift_amount = chamber_height * self.smooth_transition(lift_progress)
                    foot[2] += lift_amount
                
            elif phase < 0.9:
                # Extension phase (shorter duration)
                progress = (phase - 0.825) / 0.075
                smooth_prog = self.smooth_transition(progress)
                
                retraction_decay = 1.0 - smooth_prog
                foot[0] -= chamber_retraction * retraction_decay
                foot[0] += self.extension_forward * 0.75 * smooth_prog
                
                if progress < 0.6:
                    descent_progress = progress / 0.6
                    remaining_height = chamber_height * (1.0 - self.smooth_transition(descent_progress))
                    foot[2] += remaining_height
                
            else:
                # Stance phase
                foot[2] = max(foot[2], 0.0)
        
        return foot

    def step(self, dt):
        """Main step function called by framework"""
        self.t += dt
        phase = (self.t * self.freq) % 1.0
        
        self.update_base_motion(phase, dt)
        
        foot_positions_body = {}
        for leg_name in self.leg_names:
            foot_positions_body[leg_name] = self.compute_foot_position_body_frame(leg_name, phase)
        
        return foot_positions_body, self.root_pos, self.root_quat, self.vel_world, self.omega_world