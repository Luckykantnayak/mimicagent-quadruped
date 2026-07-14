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
    
    FIXED: Chambering now correctly retracts foot BACKWARD (not forward) to avoid joint limits
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Chamber and extension parameters - corrected geometry
        self.rear_chamber_height = 0.07  # Further reduced to minimize joint angle demands
        self.front_chamber_height = 0.09  # Moderate height for front legs
        self.rear_chamber_retract_x = -0.05  # NEGATIVE: pulls foot BACKWARD during chamber
        self.front_chamber_retract_x = -0.03  # Front legs also retract backward
        self.chamber_retract_y = 0.025  # Minimal lateral retraction
        
        # Extension parameters: forward reach from base position
        self.rear_extension_x = 0.12  # Forward extension from base position
        self.front_extension_x = 0.10  # Moderate forward extension for front legs
        
        # Base velocity parameters
        self.vx_coast = 0.4  # Coasting forward velocity
        self.vx_surge_rear = 1.1  # Explosive surge during rear leg extension
        self.vx_surge_front = 0.65  # Moderate surge during front leg extension
        self.vz_impulse = 0.08  # Minimal upward impulse
        
        # Pitch rate parameters
        self.pitch_rate_rear_kick = 0.2  # Moderate nose-up during rear leg extension
        self.pitch_rate_correction = -0.12  # Gentle pitch correction
        
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
        """
        vx = self.vx_coast
        vz = 0.0
        pitch_rate = 0.0
        
        # RL chamber (0.0-0.18): coast
        if 0.0 <= phase < 0.18:
            vx = self.vx_coast
            
        # RL extend (0.18-0.32): explosive surge
        elif 0.18 <= phase < 0.32:
            progress = (phase - 0.18) / 0.14
            vx = self.vx_surge_rear
            vz = self.vz_impulse * np.sin(np.pi * progress)
            pitch_rate = self.pitch_rate_rear_kick * (1.0 - progress)
            
        # RR chamber (0.32-0.50): coast with pitch correction
        elif 0.32 <= phase < 0.50:
            vx = self.vx_coast
            progress = (phase - 0.32) / 0.18
            pitch_rate = self.pitch_rate_correction * (1.0 - progress)
            
        # RR extend (0.50-0.64): explosive surge
        elif 0.50 <= phase < 0.64:
            progress = (phase - 0.50) / 0.14
            vx = self.vx_surge_rear
            vz = self.vz_impulse * np.sin(np.pi * progress)
            pitch_rate = self.pitch_rate_rear_kick * (1.0 - progress)
            
        # FL chamber-extend (0.64-0.78): moderate forward motion
        elif 0.64 <= phase < 0.78:
            progress = (phase - 0.64) / 0.14
            vx = self.vx_surge_front
            pitch_rate = self.pitch_rate_correction * 0.5 * (1.0 - progress)
            
        # FR chamber-extend (0.78-0.92): moderate forward motion
        elif 0.78 <= phase < 0.92:
            progress = (phase - 0.78) / 0.14
            vx = self.vx_surge_front
            pitch_rate = self.pitch_rate_correction * 0.5 * (1.0 - progress)
            
        # Neutral transition (0.92-1.0): coast
        else:
            vx = self.vx_coast * 0.75
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
        
        KEY FIX: Chambering now retracts foot BACKWARD (negative x offset) and upward,
        then extension kicks FORWARD from this retracted position.
        """
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is rear or front
        is_rear = leg_name.startswith('R')
        
        # Select parameters based on leg type
        if is_rear:
            chamber_height = self.rear_chamber_height
            chamber_retract_x = self.rear_chamber_retract_x  # NEGATIVE value
            extension_x = self.rear_extension_x
        else:
            chamber_height = self.front_chamber_height
            chamber_retract_x = self.front_chamber_retract_x  # NEGATIVE value
            extension_x = self.front_extension_x
        
        # Lateral offset for chambering
        chamber_y_factor = -1.0 if leg_name.endswith('L') else 1.0
        
        # RL: chamber (0.0-0.18), extend (0.18-0.32), stance (0.32-1.0)
        if leg_name.startswith('RL'):
            if 0.0 <= phase < 0.18:
                # Chamber: retract BACKWARD and lift upward
                progress = phase / 0.18
                smooth_progress = self._smooth_step(progress)
                foot = base_foot.copy()
                foot[0] += chamber_retract_x * smooth_progress  # Negative: moves backward
                foot[1] -= chamber_y_factor * self.chamber_retract_y * smooth_progress
                foot[2] += chamber_height * smooth_progress
                
            elif 0.18 <= phase < 0.28:
                # Extension swing: from chambered (backward) position forward
                progress = (phase - 0.18) / 0.10
                smooth_progress = self._smooth_step(progress)
                arc_progress = np.sin(np.pi * progress)
                
                foot = base_foot.copy()
                # Interpolate from chambered (backward) to extended (forward) position
                x_offset = chamber_retract_x * (1.0 - smooth_progress) + extension_x * smooth_progress
                foot[0] += x_offset
                foot[1] -= chamber_y_factor * self.chamber_retract_y * (1.0 - smooth_progress)
                foot[2] += chamber_height * (1.0 - smooth_progress) + 0.03 * arc_progress
                
            elif 0.28 <= phase < 0.32:
                # Ground contact: foot at extended forward position
                progress = (phase - 0.28) / 0.04
                foot = base_foot.copy()
                foot[0] += extension_x * (1.0 - 0.05 * progress)
                
            else:
                # Stance: foot moves rearward in body frame as base advances
                stance_progress = (phase - 0.32) / 0.68
                foot = base_foot.copy()
                # Foot moves from forward extended position to rearward
                foot[0] += extension_x * (1.0 - stance_progress * 1.2)
        
        # RR: stance (0.0-0.32), chamber (0.32-0.50), extend (0.50-0.64), stance (0.64-1.0)
        elif leg_name.startswith('RR'):
            if phase < 0.32:
                # Stance: continuing from previous cycle
                stance_progress = phase / 0.32
                foot = base_foot.copy()
                foot[0] += extension_x * (1.0 - stance_progress * 1.2)
                
            elif 0.32 <= phase < 0.50:
                # Chamber: retract BACKWARD and lift
                progress = (phase - 0.32) / 0.18
                smooth_progress = self._smooth_step(progress)
                foot = base_foot.copy()
                foot[0] += chamber_retract_x * smooth_progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * smooth_progress
                foot[2] += chamber_height * smooth_progress
                
            elif 0.50 <= phase < 0.60:
                # Extension swing
                progress = (phase - 0.50) / 0.10
                smooth_progress = self._smooth_step(progress)
                arc_progress = np.sin(np.pi * progress)
                
                foot = base_foot.copy()
                x_offset = chamber_retract_x * (1.0 - smooth_progress) + extension_x * smooth_progress
                foot[0] += x_offset
                foot[1] -= chamber_y_factor * self.chamber_retract_y * (1.0 - smooth_progress)
                foot[2] += chamber_height * (1.0 - smooth_progress) + 0.03 * arc_progress
                
            elif 0.60 <= phase < 0.64:
                # Ground contact
                progress = (phase - 0.60) / 0.04
                foot = base_foot.copy()
                foot[0] += extension_x * (1.0 - 0.05 * progress)
                
            else:
                # Stance
                stance_progress = (phase - 0.64) / 0.36
                foot = base_foot.copy()
                foot[0] += extension_x * (1.0 - stance_progress * 1.2)
        
        # FL: stance (0.0-0.64), chamber-extend (0.64-0.78), stance (0.78-1.0)
        elif leg_name.startswith('FL'):
            if phase < 0.64:
                # Stance
                stance_progress = phase / 0.64
                foot = base_foot.copy()
                foot[0] += extension_x * (1.0 - stance_progress * 1.1)
                
            elif 0.64 <= phase < 0.73:
                # Combined chamber-extend
                progress = (phase - 0.64) / 0.09
                smooth_progress = self._smooth_step(progress)
                
                foot = base_foot.copy()
                # Brief backward retraction followed by forward extension
                if progress < 0.4:
                    chamber_prog = progress / 0.4
                    foot[0] += chamber_retract_x * chamber_prog
                    foot[2] += chamber_height * chamber_prog
                    foot[1] -= chamber_y_factor * self.chamber_retract_y * 0.5 * chamber_prog
                else:
                    extend_prog = (progress - 0.4) / 0.6
                    extend_smooth = self._smooth_step(extend_prog)
                    x_offset = chamber_retract_x * (1.0 - extend_smooth) + extension_x * extend_smooth
                    foot[0] += x_offset
                    foot[2] += chamber_height * (1.0 - extend_smooth)
                    foot[1] -= chamber_y_factor * self.chamber_retract_y * 0.5 * (1.0 - extend_smooth)
                
            elif 0.73 <= phase < 0.78:
                # Plant
                progress = (phase - 0.73) / 0.05
                foot = base_foot.copy()
                foot[0] += extension_x * (1.0 - 0.04 * progress)
                
            else:
                # Stance
                stance_progress = (phase - 0.78) / 0.22
                foot = base_foot.copy()
                foot[0] += extension_x * (1.0 - stance_progress * 1.1)
        
        # FR: stance (0.0-0.78), chamber-extend (0.78-0.92), stance (0.92-1.0)
        elif leg_name.startswith('FR'):
            if phase < 0.78:
                # Stance
                stance_progress = phase / 0.78
                foot = base_foot.copy()
                foot[0] += extension_x * (1.0 - stance_progress * 1.1)
                
            elif 0.78 <= phase < 0.87:
                # Combined chamber-extend
                progress = (phase - 0.78) / 0.09
                smooth_progress = self._smooth_step(progress)
                
                foot = base_foot.copy()
                if progress < 0.4:
                    chamber_prog = progress / 0.4
                    foot[0] += chamber_retract_x * chamber_prog
                    foot[2] += chamber_height * chamber_prog
                    foot[1] -= chamber_y_factor * self.chamber_retract_y * 0.5 * chamber_prog
                else:
                    extend_prog = (progress - 0.4) / 0.6
                    extend_smooth = self._smooth_step(extend_prog)
                    x_offset = chamber_retract_x * (1.0 - extend_smooth) + extension_x * extend_smooth
                    foot[0] += x_offset
                    foot[2] += chamber_height * (1.0 - extend_smooth)
                    foot[1] -= chamber_y_factor * self.chamber_retract_y * 0.5 * (1.0 - extend_smooth)
                
            elif 0.87 <= phase < 0.92:
                # Plant
                progress = (phase - 0.87) / 0.05
                foot = base_foot.copy()
                foot[0] += extension_x * (1.0 - 0.04 * progress)
                
            else:
                # Neutral stance
                stance_progress = (phase - 0.92) / 0.08
                foot = base_foot.copy()
                foot[0] += extension_x * (1.0 - stance_progress * 1.1)
        
        return foot

    def _smooth_step(self, t):
        """Smooth step function for smoother transitions (3t^2 - 2t^3)."""
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)