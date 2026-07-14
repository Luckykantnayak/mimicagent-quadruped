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
        self.compression_depth = 0.008  # Minimal leg compression during loading
        
        # Base motion parameters - rebalanced for adequate height clearance
        self.forward_velocity = 0.6  # Forward velocity during thrust
        self.vertical_velocity_thrust = 0.65  # Increased upward velocity during thrust
        self.vertical_velocity_liftoff = 0.50  # Sustained upward velocity during front liftoff
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
        Vertical velocity profile rebalanced to ensure adequate height clearance.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        if phase < 0.15:
            # Rear liftoff: front legs thrust, strong upward velocity with minimal decay
            progress = phase / 0.15
            vx = self.forward_velocity
            # Strong upward thrust with minimal decay to ensure height gain
            vz = self.vertical_velocity_thrust * (1.0 - progress * 0.15)
            pitch_rate = self.pitch_rate_thrust * (1.0 - progress)
            
        elif phase < 0.3:
            # Front liftoff: sustained upward velocity to ensure apex height
            progress = (phase - 0.15) / 0.15
            vx = self.forward_velocity * (1.0 - progress * 0.2)
            # Sustained upward velocity with gradual decay
            vz = self.vertical_velocity_liftoff * (1.0 - progress * 0.35)
            pitch_rate = self.pitch_rate_level * progress
            
        elif phase < 0.5:
            # Flight apex: smooth transition from ascent to gentle descent
            progress = (phase - 0.3) / 0.2
            vx = self.forward_velocity * 0.8 * (1.0 - progress * 0.25)
            # Smooth transition through apex with gentle descent initiation
            apex_vz_start = self.vertical_velocity_liftoff * 0.65
            apex_vz_end = -0.20  # Gentle initial descent
            vz = apex_vz_start + (apex_vz_end - apex_vz_start) * (progress ** 1.5)
            pitch_rate = 0.0
            
        elif phase < 0.65:
            # Rear landing: moderate controlled descent
            progress = (phase - 0.5) / 0.15
            vx = self.forward_velocity * 0.6 * (1.0 - progress * 0.4)
            # Moderate descent velocity for controlled landing
            vz = -0.20 - progress * 0.12
            pitch_rate = self.pitch_rate_landing * (1.0 - progress * 0.5)
            
        elif phase < 0.8:
            # Front landing: continued moderate descent
            progress = (phase - 0.65) / 0.15
            vx = self.forward_velocity * 0.35 * (1.0 - progress)
            # Controlled descent with gradual deceleration
            vz = -0.32 + progress * 0.20
            pitch_rate = self.pitch_rate_correct * progress
            
        else:
            # Compression and loading: gentle settling with minimal downward velocity
            progress = (phase - 0.8) / 0.2
            vx = 0.08 * (1.0 - progress)
            # Minimal settling velocity
            vz = -0.12 * (1.0 - progress * 0.85)
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
        Compression is minimal and applied only during final loading phase.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_rear = leg_name.startswith('R')
        is_front = leg_name.startswith('F')
        
        if is_rear:
            # Rear legs: lift [0.0, 0.15], flight [0.15, 0.5], land [0.5, 0.65], stance [0.65, 1.0]
            if phase < 0.15:
                # Liftoff phase: smooth upward trajectory
                progress = phase / 0.15
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
                # Landing phase: contact at nominal height (no compression)
                progress = (phase - 0.5) / 0.15
                foot[0] += self.forward_reach * 0.5
                # No compression during landing contact
                
            else:
                # Stance phase: minimal compression only in final loading phase
                progress = (phase - 0.65) / 0.35
                foot[0] += self.forward_reach * 0.5
                if progress > 0.7:
                    # Minimal compression in final loading preparation
                    compress_progress = (progress - 0.7) / 0.3
                    foot[2] -= self.compression_depth * compress_progress
        
        elif is_front:
            # Front legs: stance [0.0, 0.15], liftoff [0.15, 0.3], flight [0.3, 0.65], land [0.65, 0.8], stance [0.8, 1.0]
            if phase < 0.15:
                # Stance phase: providing thrust with minimal motion
                progress = phase / 0.15
                foot[0] -= self.forward_reach * 0.2 * progress
                
            elif phase < 0.3:
                # Liftoff phase: smooth upward trajectory
                progress = (phase - 0.15) / 0.15
                lift_curve = np.sin(progress * np.pi / 2)
                foot[2] += self.tuck_height * lift_curve
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
                # Landing phase: contact at nominal height (no compression)
                progress = (phase - 0.65) / 0.15
                foot[0] += self.forward_reach * 0.5
                # No compression during landing contact
                
            else:
                # Stance phase: minimal compression in final loading
                progress = (phase - 0.8) / 0.2
                foot[0] += self.forward_reach * 0.5
                # Minimal progressive compression for energy loading
                foot[2] -= self.compression_depth * progress
        
        return foot