from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_WAVE_HOP_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Wave hop forward motion: rear legs lift first, then front legs,
    creating a wave-like hopping motion from rear to front.
    
    Phase structure:
      [0.0, 0.15]: rear_liftoff - rear legs lift, front legs thrust
      [0.15, 0.3]: front_liftoff - front legs lift, entering flight
      [0.3, 0.5]: flight_apex - all legs airborne
      [0.5, 0.65]: rear_landing - rear legs contact first
      [0.65, 0.8]: front_landing - front legs contact
      [0.8, 1.0]: compression_and_loading - all legs compress
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.hop_height = 0.15  # Maximum z lift during flight
        self.tuck_height = 0.12  # How much legs tuck during flight
        self.forward_reach = 0.08  # Forward displacement during hop
        self.compression_depth = 0.04  # Leg compression during loading
        
        # Base motion parameters
        self.forward_velocity = 0.6  # Forward velocity during thrust
        self.vertical_velocity_thrust = 0.8  # Upward velocity during thrust
        self.vertical_velocity_apex = 0.4  # Upward velocity at liftoff, decreasing
        self.pitch_rate_thrust = 0.8  # Nose-up rate during rear liftoff
        self.pitch_rate_level = -0.6  # Correction rate to level
        self.pitch_rate_landing = -0.5  # Nose-down rate during rear landing
        self.pitch_rate_correct = 0.4  # Correction during front landing
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Wave pattern creates sequential pitch oscillations.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        if phase < 0.15:
            # Rear liftoff: front legs thrust, forward and upward velocity
            progress = phase / 0.15
            vx = self.forward_velocity
            vz = self.vertical_velocity_thrust * (1.0 - progress * 0.3)
            pitch_rate = self.pitch_rate_thrust * (1.0 - progress)
            
        elif phase < 0.3:
            # Front liftoff: continue upward, level pitch
            progress = (phase - 0.15) / 0.15
            vx = self.forward_velocity * (1.0 - progress * 0.3)
            vz = self.vertical_velocity_apex * (1.0 - progress * 0.5)
            pitch_rate = self.pitch_rate_level * progress
            
        elif phase < 0.5:
            # Flight apex: ballistic trajectory
            progress = (phase - 0.3) / 0.2
            vx = self.forward_velocity * 0.7 * (1.0 - progress * 0.3)
            # Vertical velocity transitions from up to down through apex
            vz = self.vertical_velocity_apex * 0.5 * (1.0 - progress) - 0.3 * progress
            pitch_rate = 0.0
            
        elif phase < 0.65:
            # Rear landing: decelerate, nose down
            progress = (phase - 0.5) / 0.15
            vx = self.forward_velocity * 0.4 * (1.0 - progress * 0.5)
            vz = -0.5 - progress * 0.3
            pitch_rate = self.pitch_rate_landing * (1.0 - progress * 0.5)
            
        elif phase < 0.8:
            # Front landing: complete deceleration, level pitch
            progress = (phase - 0.65) / 0.15
            vx = self.forward_velocity * 0.2 * (1.0 - progress)
            vz = -0.6 * (1.0 - progress * 0.5)
            pitch_rate = self.pitch_rate_correct * progress
            
        else:
            # Compression and loading: settle into stance
            progress = (phase - 0.8) / 0.2
            vx = 0.05 * (1.0 - progress)
            vz = -0.2 * (1.0 - progress * 0.7)
            pitch_rate = 0.0
        
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
        Compute foot trajectory in body frame.
        Rear legs (RL, RR) lift at phase 0.0, land at phase 0.5.
        Front legs (FL, FR) lift at phase 0.15, land at phase 0.65.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_rear = leg_name.startswith('R')
        is_front = leg_name.startswith('F')
        
        if is_rear:
            # Rear legs: lift [0.0, 0.15], flight [0.15, 0.5], land [0.5, 0.65], stance [0.65, 1.0]
            if phase < 0.15:
                # Liftoff phase
                progress = phase / 0.15
                # Smooth liftoff with sinusoidal profile
                lift_curve = np.sin(progress * np.pi / 2)
                foot[2] += self.tuck_height * lift_curve
                foot[0] += self.forward_reach * 0.1 * lift_curve
                
            elif phase < 0.5:
                # Flight phase: tucked, transitioning to landing extension
                progress = (phase - 0.15) / 0.35
                if progress < 0.5:
                    # Maintain tuck
                    foot[2] += self.tuck_height
                    foot[0] += self.forward_reach * 0.1
                else:
                    # Begin extending for landing
                    extend_progress = (progress - 0.5) / 0.5
                    tuck_factor = 1.0 - extend_progress
                    foot[2] += self.tuck_height * tuck_factor
                    foot[0] += self.forward_reach * (0.1 + 0.4 * extend_progress)
                    
            elif phase < 0.65:
                # Landing phase: contact and compress
                progress = (phase - 0.5) / 0.15
                # Smooth landing compression
                compress_curve = np.sin(progress * np.pi / 2)
                foot[0] += self.forward_reach * 0.5
                foot[2] -= self.compression_depth * 0.5 * compress_curve
                
            else:
                # Stance phase: compressed and loading
                progress = (phase - 0.65) / 0.35
                if progress < 0.4:
                    # Continue compression as front lands
                    compress_progress = progress / 0.4
                    foot[0] += self.forward_reach * 0.5
                    foot[2] -= self.compression_depth * (0.5 + 0.5 * compress_progress)
                else:
                    # Maximum compression, loading energy
                    foot[0] += self.forward_reach * 0.5
                    foot[2] -= self.compression_depth
        
        elif is_front:
            # Front legs: stance [0.0, 0.15], liftoff [0.15, 0.3], flight [0.3, 0.65], land [0.65, 0.8], stance [0.8, 1.0]
            if phase < 0.15:
                # Stance phase: providing thrust
                progress = phase / 0.15
                # Slight extension during thrust
                foot[2] += self.compression_depth * 0.3 * progress
                foot[0] -= self.forward_reach * 0.2 * progress
                
            elif phase < 0.3:
                # Liftoff phase
                progress = (phase - 0.15) / 0.15
                lift_curve = np.sin(progress * np.pi / 2)
                foot[2] += self.compression_depth * 0.3 + self.tuck_height * lift_curve
                foot[0] -= self.forward_reach * 0.2 * (1.0 - lift_curve)
                
            elif phase < 0.65:
                # Flight phase: tucked, then extending for landing
                progress = (phase - 0.3) / 0.35
                if progress < 0.6:
                    # Maintain tuck
                    foot[2] += self.tuck_height
                else:
                    # Begin extending for landing
                    extend_progress = (progress - 0.6) / 0.4
                    tuck_factor = 1.0 - extend_progress
                    foot[2] += self.tuck_height * tuck_factor
                    foot[0] += self.forward_reach * 0.5 * extend_progress
                    
            elif phase < 0.8:
                # Landing phase: contact and compress
                progress = (phase - 0.65) / 0.15
                compress_curve = np.sin(progress * np.pi / 2)
                foot[0] += self.forward_reach * 0.5
                foot[2] -= self.compression_depth * 0.5 * compress_curve
                
            else:
                # Stance phase: compressed and loading
                progress = (phase - 0.8) / 0.2
                foot[0] += self.forward_reach * 0.5
                # Gradually compress to maximum
                foot[2] -= self.compression_depth * (0.5 + 0.5 * progress)
        
        return foot