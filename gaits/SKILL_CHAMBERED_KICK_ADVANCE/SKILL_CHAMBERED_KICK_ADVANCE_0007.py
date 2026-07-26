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
        
        # Chamber and extension parameters
        self.chamber_height = 0.15  # Height during chambered position
        self.chamber_retract_x = 0.08  # Forward retraction during chamber
        self.chamber_retract_y = 0.05  # Lateral retraction toward body center
        
        # Extension parameters (forward reach from chamber)
        self.rear_extension_x = 0.2  # Rear leg extension distance
        self.front_extension_x = 0.15  # Front leg extension distance
        self.extension_z_start = 0.1  # Extension starts from chambered height
        
        # Base velocity parameters
        self.vx_coast = 0.4  # Coasting forward velocity
        self.vx_surge_rear = 1.2  # Explosive surge during rear leg extension
        self.vx_surge_front = 0.7  # Moderate surge during front leg extension
        self.vz_impulse = 0.15  # Upward impulse during rear leg push
        
        # Pitch rate parameters
        self.pitch_rate_rear_kick = 0.3  # Nose-up during rear leg extension
        self.pitch_rate_correction = -0.2  # Pitch correction rate
        
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
        
        Rear leg extensions (RL: 0.15-0.3, RR: 0.45-0.6) generate explosive forward surges.
        Front leg extensions (FL: 0.6-0.75, FR: 0.75-0.9) maintain forward momentum.
        Other phases coast with moderate velocity.
        """
        vx = self.vx_coast
        vz = 0.0
        pitch_rate = 0.0
        
        # RL chamber (0.0-0.15): coast
        if 0.0 <= phase < 0.15:
            vx = self.vx_coast
            
        # RL extend (0.15-0.3): explosive surge
        elif 0.15 <= phase < 0.3:
            progress = (phase - 0.15) / 0.15
            vx = self.vx_surge_rear
            # Upward impulse peaks mid-extension then decays
            vz = self.vz_impulse * np.sin(np.pi * progress)
            # Pitch up during extension
            pitch_rate = self.pitch_rate_rear_kick * (1.0 - progress)
            
        # RR chamber (0.3-0.45): coast with pitch correction
        elif 0.3 <= phase < 0.45:
            vx = self.vx_coast
            pitch_rate = self.pitch_rate_correction
            
        # RR extend (0.45-0.6): explosive surge
        elif 0.45 <= phase < 0.6:
            progress = (phase - 0.45) / 0.15
            vx = self.vx_surge_rear
            vz = self.vz_impulse * np.sin(np.pi * progress)
            pitch_rate = self.pitch_rate_rear_kick * (1.0 - progress)
            
        # FL chamber-extend (0.6-0.75): moderate forward motion
        elif 0.6 <= phase < 0.75:
            progress = (phase - 0.6) / 0.15
            vx = self.vx_surge_front
            # Pitch correction as front leg extends
            pitch_rate = self.pitch_rate_correction * (1.0 - progress)
            
        # FR chamber-extend (0.75-0.9): moderate forward motion
        elif 0.75 <= phase < 0.9:
            progress = (phase - 0.75) / 0.15
            vx = self.vx_surge_front
            pitch_rate = self.pitch_rate_correction * (1.0 - progress)
            
        # Neutral transition (0.9-1.0): coast, zero angular rates
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
        
        Sequence: RL (0.0-0.3) → RR (0.3-0.6) → FL (0.6-0.75) → FR (0.75-0.9)
        Each leg: chamber (lift + retract) → extend (kick forward-down) → stance
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is rear or front for parameter selection
        is_rear = leg_name.startswith('R')
        extension_x = self.rear_extension_x if is_rear else self.front_extension_x
        
        # Lateral offset for chambering (retract toward body centerline)
        chamber_y_factor = -1.0 if leg_name.endswith('L') else 1.0
        
        # RL: chamber (0.0-0.15), extend (0.15-0.3), stance (0.3-1.0)
        if leg_name.startswith('RL'):
            if 0.0 <= phase < 0.15:
                # Chamber: retract and lift
                progress = phase / 0.15
                smooth_progress = self._smooth_step(progress)
                foot[0] += self.chamber_retract_x * smooth_progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * smooth_progress
                foot[2] += self.chamber_height * smooth_progress
                
            elif 0.15 <= phase < 0.25:
                # Extension swing: ballistic arc from chamber to forward position
                progress = (phase - 0.15) / 0.1
                arc_progress = np.sin(np.pi * progress)
                # Start from chambered position
                foot[0] += self.chamber_retract_x + (extension_x - self.chamber_retract_x) * progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * (1.0 - progress)
                foot[2] += self.chamber_height * (1.0 - progress) + 0.05 * arc_progress
                
            elif 0.25 <= phase < 0.3:
                # Ground contact and push: foot plants forward
                progress = (phase - 0.25) / 0.05
                foot[0] += extension_x * (1.0 - 0.1 * progress)
                
            else:
                # Stance: foot moves rearward in body frame as base advances
                stance_progress = (phase - 0.3) / 0.7
                foot[0] += extension_x * (1.0 - stance_progress)
        
        # RR: stance (0.0-0.3), chamber (0.3-0.45), extend (0.45-0.6), stance (0.6-1.0)
        elif leg_name.startswith('RR'):
            if phase < 0.3:
                # Stance from previous cycle
                stance_progress = phase / 0.3
                foot[0] += extension_x * (1.0 - stance_progress)
                
            elif 0.3 <= phase < 0.45:
                # Chamber
                progress = (phase - 0.3) / 0.15
                smooth_progress = self._smooth_step(progress)
                foot[0] += self.chamber_retract_x * smooth_progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * smooth_progress
                foot[2] += self.chamber_height * smooth_progress
                
            elif 0.45 <= phase < 0.55:
                # Extension swing
                progress = (phase - 0.45) / 0.1
                arc_progress = np.sin(np.pi * progress)
                foot[0] += self.chamber_retract_x + (extension_x - self.chamber_retract_x) * progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * (1.0 - progress)
                foot[2] += self.chamber_height * (1.0 - progress) + 0.05 * arc_progress
                
            elif 0.55 <= phase < 0.6:
                # Ground contact and push
                progress = (phase - 0.55) / 0.05
                foot[0] += extension_x * (1.0 - 0.1 * progress)
                
            else:
                # Stance
                stance_progress = (phase - 0.6) / 0.4
                foot[0] += extension_x * (1.0 - stance_progress)
        
        # FL: stance (0.0-0.6), chamber-extend (0.6-0.75), stance (0.75-1.0)
        elif leg_name.startswith('FL'):
            if phase < 0.6:
                # Stance
                stance_progress = phase / 0.6
                foot[0] += extension_x * (1.0 - stance_progress)
                
            elif 0.6 <= phase < 0.68:
                # Rapid chamber-extend combined
                progress = (phase - 0.6) / 0.08
                smooth_progress = self._smooth_step(progress)
                # Chamber up
                foot[2] += self.chamber_height * 0.7 * np.sin(np.pi * progress)
                # Extend forward simultaneously
                foot[0] += extension_x * smooth_progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * 0.5 * np.sin(np.pi * progress)
                
            elif 0.68 <= phase < 0.75:
                # Plant and stabilize
                progress = (phase - 0.68) / 0.07
                foot[0] += extension_x * (1.0 - 0.05 * progress)
                
            else:
                # Stance
                stance_progress = (phase - 0.75) / 0.25
                foot[0] += extension_x * (1.0 - stance_progress)
        
        # FR: stance (0.0-0.75), chamber-extend (0.75-0.9), stance (0.9-1.0)
        elif leg_name.startswith('FR'):
            if phase < 0.75:
                # Stance
                stance_progress = phase / 0.75
                foot[0] += extension_x * (1.0 - stance_progress)
                
            elif 0.75 <= phase < 0.83:
                # Rapid chamber-extend combined
                progress = (phase - 0.75) / 0.08
                smooth_progress = self._smooth_step(progress)
                foot[2] += self.chamber_height * 0.7 * np.sin(np.pi * progress)
                foot[0] += extension_x * smooth_progress
                foot[1] -= chamber_y_factor * self.chamber_retract_y * 0.5 * np.sin(np.pi * progress)
                
            elif 0.83 <= phase < 0.9:
                # Plant and stabilize
                progress = (phase - 0.83) / 0.07
                foot[0] += extension_x * (1.0 - 0.05 * progress)
                
            else:
                # Stance (neutral phase)
                stance_progress = (phase - 0.9) / 0.1
                foot[0] += extension_x * (1.0 - stance_progress)
        
        return foot

    def _smooth_step(self, t):
        """Smooth step function for smoother transitions (3t^2 - 2t^3)."""
        return t * t * (3.0 - 2.0 * t)