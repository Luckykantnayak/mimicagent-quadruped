from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CHAMBERED_KICK_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Chambered kick advance gait: sequential leg chamber-extend cycles.
    
    Each leg chambers (retracts close to body) then explosively extends forward-downward
    to propel the base forward. Sequence: RL → RR → FL → FR.
    
    - Rear legs generate stronger forward propulsion impulses
    - Front legs extend the stride with less explosive force
    - Tripod support maintained throughout most of the cycle
    - Base forward velocity surges during extension phases
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Chamber and extension parameters - REDUCED for rear legs to avoid knee limit violations
        self.rear_chamber_height = 0.10  # Reduced from 0.15 to avoid excessive knee flexion
        self.front_chamber_height = 0.12  # Slightly reduced for consistency
        self.rear_chamber_retract_x = 0.04  # Reduced from 0.08 for rear legs
        self.front_chamber_retract_x = 0.06  # Front legs can handle more retraction
        self.chamber_retract_y = 0.03  # Reduced lateral retraction to ease knee constraints
        
        # Extension parameters (forward reach from chamber)
        self.rear_extension_x = 0.18  # Slightly reduced from 0.2 for smoother kinematics
        self.front_extension_x = 0.14  # Slightly reduced from 0.15
        
        # Base velocity parameters
        self.vx_coast = 0.4  # Coasting forward velocity
        self.vx_surge_rear = 1.2  # Explosive surge during rear leg extension
        self.vx_surge_front = 0.7  # Moderate surge during front leg extension
        self.vz_impulse = 0.12  # Reduced upward impulse to avoid body lifting too much
        
        # Pitch rate parameters
        self.pitch_rate_rear_kick = 0.25  # Slightly reduced nose-up during rear leg extension
        self.pitch_rate_correction = -0.15  # Gentler pitch correction
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocities based on which leg is extending.
        
        Rear leg extensions (RL: 0.17-0.32, RR: 0.49-0.64) generate explosive forward surges.
        Front leg extensions (FL: 0.65-0.78, FR: 0.82-0.94) maintain forward momentum.
        Other phases coast with moderate velocity.
        """
        vx = self.vx_coast
        vz = 0.0
        pitch_rate = 0.0
        
        # RL chamber (0.0-0.17): coast - EXTENDED DURATION
        if 0.0 <= phase < 0.17:
            vx = self.vx_coast
            
        # RL extend (0.17-0.32): explosive surge - EXTENDED DURATION
        elif 0.17 <= phase < 0.32:
            progress = (phase - 0.17) / 0.15
            vx = self.vx_surge_rear
            # Upward impulse peaks mid-extension then decays
            vz = self.vz_impulse * np.sin(np.pi * progress)
            # Pitch up during extension
            pitch_rate = self.pitch_rate_rear_kick * (1.0 - progress)
            
        # RR chamber (0.32-0.49): coast with pitch correction - EXTENDED DURATION
        elif 0.32 <= phase < 0.49:
            vx = self.vx_coast
            pitch_rate = self.pitch_rate_correction
            
        # RR extend (0.49-0.64): explosive surge - EXTENDED DURATION
        elif 0.49 <= phase < 0.64:
            progress = (phase - 0.49) / 0.15
            vx = self.vx_surge_rear
            vz = self.vz_impulse * np.sin(np.pi * progress)
            pitch_rate = self.pitch_rate_rear_kick * (1.0 - progress)
            
        # FL chamber-extend (0.64-0.78): moderate forward motion
        elif 0.64 <= phase < 0.78:
            progress = (phase - 0.64) / 0.14
            vx = self.vx_surge_front
            # Pitch correction as front leg extends
            pitch_rate = self.pitch_rate_correction * (1.0 - progress)
            
        # FR chamber-extend (0.78-0.92): moderate forward motion
        elif 0.78 <= phase < 0.92:
            progress = (phase - 0.78) / 0.14
            vx = self.vx_surge_front
            pitch_rate = self.pitch_rate_correction * (1.0 - progress)
            
        # Neutral transition (0.92-1.0): coast, zero angular rates
        else:
            vx = self.vx_coast * 0.7  # Slight deceleration for smooth transition
            pitch_rate = 0.0
        
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in BODY frame based on phase and leg sequence.
        
        Sequence: RL (0.0-0.32) → RR (0.32-0.64) → FL (0.64-0.78) → FR (0.78-0.92)
        Each leg: chamber (lift + retract) → extend (kick forward-down) → stance
        
        Rear legs use reduced chamber heights and retraction distances to avoid knee limit violations.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is rear or front for parameter selection
        is_rear = leg_name.startswith('R')
        
        # Select parameters based on leg type
        if is_rear:
            chamber_height = self.rear_chamber_height
            chamber_retract_x = self.rear_chamber_retract_x
            extension_x = self.rear_extension_x
        else:
            chamber_height = self.front_chamber_height
            chamber_retract_x = self.front_chamber_retract_x
            extension_x = self.front_extension_x
        
        # Lateral offset for chambering (retract toward body centerline)
        chamber_y_factor = -1.0 if leg_name.endswith('L') else 1.0
        
        # RL: chamber (0.0-0.17), extend (0.17-0.32), stance (0.32-1.0) - EXTENDED DURATIONS
        if leg_name.startswith('RL'):
            if 0.0 <= phase < 0.17:
                # Chamber: retract and lift - LONGER DURATION for smoother knee motion
                progress = phase / 0.17
                smooth_progress = self._smooth_step(progress)
                foot[0] += chamber_retract_x * smooth_progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * smooth_progress
                foot[2] += chamber_height * smooth_progress
                
            elif 0.17 <= phase < 0.27:
                # Extension swing: ballistic arc from chamber to forward position - LONGER DURATION
                progress = (phase - 0.17) / 0.10
                arc_progress = np.sin(np.pi * progress)
                smooth_progress = self._smooth_step(progress)
                # Start from chambered position
                foot[0] += chamber_retract_x + (extension_x - chamber_retract_x) * smooth_progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * (1.0 - smooth_progress)
                foot[2] += chamber_height * (1.0 - smooth_progress) + 0.04 * arc_progress
                
            elif 0.27 <= phase < 0.32:
                # Ground contact and push: foot plants forward
                progress = (phase - 0.27) / 0.05
                foot[0] += extension_x * (1.0 - 0.08 * progress)
                
            else:
                # Stance: foot moves rearward in body frame as base advances
                stance_progress = (phase - 0.32) / 0.68
                foot[0] += extension_x * (1.0 - stance_progress * 0.9)
        
        # RR: stance (0.0-0.32), chamber (0.32-0.49), extend (0.49-0.64), stance (0.64-1.0) - EXTENDED DURATIONS
        elif leg_name.startswith('RR'):
            if phase < 0.32:
                # Stance from previous cycle
                stance_progress = phase / 0.32
                foot[0] += extension_x * (1.0 - stance_progress * 0.9)
                
            elif 0.32 <= phase < 0.49:
                # Chamber - LONGER DURATION
                progress = (phase - 0.32) / 0.17
                smooth_progress = self._smooth_step(progress)
                foot[0] += chamber_retract_x * smooth_progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * smooth_progress
                foot[2] += chamber_height * smooth_progress
                
            elif 0.49 <= phase < 0.59:
                # Extension swing - LONGER DURATION
                progress = (phase - 0.49) / 0.10
                arc_progress = np.sin(np.pi * progress)
                smooth_progress = self._smooth_step(progress)
                foot[0] += chamber_retract_x + (extension_x - chamber_retract_x) * smooth_progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * (1.0 - smooth_progress)
                foot[2] += chamber_height * (1.0 - smooth_progress) + 0.04 * arc_progress
                
            elif 0.59 <= phase < 0.64:
                # Ground contact and push
                progress = (phase - 0.59) / 0.05
                foot[0] += extension_x * (1.0 - 0.08 * progress)
                
            else:
                # Stance
                stance_progress = (phase - 0.64) / 0.36
                foot[0] += extension_x * (1.0 - stance_progress * 0.9)
        
        # FL: stance (0.0-0.64), chamber-extend (0.64-0.78), stance (0.78-1.0)
        elif leg_name.startswith('FL'):
            if phase < 0.64:
                # Stance
                stance_progress = phase / 0.64
                foot[0] += extension_x * (1.0 - stance_progress * 0.9)
                
            elif 0.64 <= phase < 0.72:
                # Rapid chamber-extend combined
                progress = (phase - 0.64) / 0.08
                smooth_progress = self._smooth_step(progress)
                # Chamber up with reduced height for front legs
                foot[2] += chamber_height * 0.8 * np.sin(np.pi * progress)
                # Extend forward simultaneously
                foot[0] += extension_x * smooth_progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * 0.6 * np.sin(np.pi * progress)
                
            elif 0.72 <= phase < 0.78:
                # Plant and stabilize
                progress = (phase - 0.72) / 0.06
                foot[0] += extension_x * (1.0 - 0.05 * progress)
                
            else:
                # Stance
                stance_progress = (phase - 0.78) / 0.22
                foot[0] += extension_x * (1.0 - stance_progress * 0.9)
        
        # FR: stance (0.0-0.78), chamber-extend (0.78-0.92), stance (0.92-1.0)
        elif leg_name.startswith('FR'):
            if phase < 0.78:
                # Stance
                stance_progress = phase / 0.78
                foot[0] += extension_x * (1.0 - stance_progress * 0.9)
                
            elif 0.78 <= phase < 0.86:
                # Rapid chamber-extend combined
                progress = (phase - 0.78) / 0.08
                smooth_progress = self._smooth_step(progress)
                foot[2] += chamber_height * 0.8 * np.sin(np.pi * progress)
                foot[0] += extension_x * smooth_progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * 0.6 * np.sin(np.pi * progress)
                
            elif 0.86 <= phase < 0.92:
                # Plant and stabilize
                progress = (phase - 0.86) / 0.06
                foot[0] += extension_x * (1.0 - 0.05 * progress)
                
            else:
                # Stance (neutral phase)
                stance_progress = (phase - 0.92) / 0.08
                foot[0] += extension_x * (1.0 - stance_progress * 0.9)
        
        return foot

    def _smooth_step(self, t):
        """Smooth step function for smoother transitions (3t^2 - 2t^3)."""
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)