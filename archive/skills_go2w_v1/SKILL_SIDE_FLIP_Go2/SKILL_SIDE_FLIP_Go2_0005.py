from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SIDE_FLIP_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Side flip kinematic motion generator.
    
    Executes a 360° roll rotation while airborne with coordinated leg repositioning.
    
    Phase structure:
      [0.0, 0.25]: Launch and initial rotation
      [0.25, 0.5]: Inverted transition
      [0.5, 0.75]: Recovery rotation
      [0.75, 1.0]: Landing and stabilization
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # 2 seconds per full flip cycle
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Flip parameters
        self.nominal_height = 0.28  # Standing height
        self.peak_altitude = 0.58  # Maximum height during flip (below 0.65 m limit)
        self.total_roll_rotation = 2 * np.pi  # 360 degrees
        
        # Roll rate tuned to complete 360° over the aerial phase
        self.peak_roll_rate = 7.5  # rad/s
        
        # Leg motion parameters (reduced for inverted phase safety)
        self.leg_retract_height = 0.16  # Reduced from 0.20 for kinematic feasibility
        self.leg_lateral_swing = 0.08  # Reduced from 0.12 for safer workspace
        
        # Define inverted-phase-specific tucked positions (relative offsets from base)
        self.inverted_tuck_offset = {
            'z': 0.10,  # Moderate tuck toward body center
            'lateral_factor': 0.5,  # Pull legs inward laterally
        }

    def compute_height_trajectory(self, phase):
        """
        Compute explicit height trajectory as function of phase.
        Returns height above ground and vertical velocity.
        """
        if phase < 0.2:
            # Launch phase: smooth rise from nominal to peak
            t = phase / 0.2
            h = self.nominal_height + (self.peak_altitude - self.nominal_height) * (3 * t**2 - 2 * t**3)
            v = (self.peak_altitude - self.nominal_height) * (6 * t - 6 * t**2) / 0.2
        elif phase < 0.7:
            # Sustained altitude during main rotation
            t = (phase - 0.2) / 0.5
            h = self.peak_altitude - 0.03 * np.sin(np.pi * t)
            v = -0.03 * np.pi * np.cos(np.pi * t) / 0.5
        else:
            # Landing phase: smooth descent back to nominal
            t = (phase - 0.7) / 0.3
            h = self.peak_altitude - (self.peak_altitude - self.nominal_height) * (3 * t**2 - 2 * t**3)
            v = -(self.peak_altitude - self.nominal_height) * (6 * t - 6 * t**2) / 0.3
        
        return h, v

    def update_base_motion(self, phase, dt):
        """
        Update base position and orientation through flip phases.
        """
        
        # Compute explicit height trajectory
        target_height, vz = self.compute_height_trajectory(phase)
        
        # Set vertical position directly to ensure bounded trajectory
        self.root_pos[2] = target_height
        
        # Roll rate profile: smooth ramp up, sustained, smooth ramp down
        if phase < 0.08:
            t = phase / 0.08
            roll_rate = self.peak_roll_rate * (3 * t**2 - 2 * t**3)
        elif phase < 0.82:
            roll_rate = self.peak_roll_rate
        else:
            t = (phase - 0.82) / 0.18
            roll_rate = self.peak_roll_rate * (1 - (3 * t**2 - 2 * t**3))
        
        # Set velocity commands
        self.vel_world = np.array([0.0, 0.0, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
        # Integrate orientation only (position set directly above)
        _, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            np.array([0.0, 0.0, 0.0]),
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame throughout flip with roll-aware frame compensation.
        
        Legs reposition through coordinated arcs with explicit handling of inverted orientation:
        - Launch: extended downward in upright frame
        - Aerial/inverted: tucked configuration with frame-compensated positioning
        - Recovery: transition back to downward extension
        - Landing: extended to nominal stance
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine leg side for symmetric motion
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        lateral_sign = 1.0 if is_left else -1.0
        
        # Estimate current roll angle (progresses approximately linearly during main rotation)
        estimated_roll = self.total_roll_rotation * np.clip((phase - 0.05) / 0.85, 0.0, 1.0)
        
        # Frame orientation factors for roll-aware positioning
        # cos(roll): 1.0 when upright, -1.0 when inverted, 0.0 at 90/270 deg
        cos_roll = np.cos(estimated_roll)
        # Inversion factor: smoothly varies from 1.0 (upright) to 0.0 (inverted)
        inversion_factor = 0.5 + 0.5 * cos_roll
        
        # Phase-dependent foot trajectory with frame-aware vertical positioning
        if phase < 0.15:
            # Launch phase: feet in nominal stance preparing for liftoff
            t = phase / 0.15
            smooth_t = 3 * t**2 - 2 * t**3
            # Slight downward extension for push-off preparation
            foot[2] = base_pos[2] - 0.01 * smooth_t
            
        elif phase < 0.25:
            # Early aerial: begin retraction with upright frame assumption
            t = (phase - 0.15) / 0.1
            smooth_t = 3 * t**2 - 2 * t**3
            
            # Initial retraction (still mostly upright)
            retract_amount = self.leg_retract_height * smooth_t * 0.4
            foot[2] = base_pos[2] + retract_amount
            
            # Begin lateral tucking
            foot[1] = base_pos[1] + lateral_sign * self.leg_lateral_swing * smooth_t * 0.3
            
        elif phase < 0.6:
            # Main aerial/inverted phase: frame-compensated tucked configuration
            t = (phase - 0.25) / 0.35
            
            # Define upright and inverted target positions
            # Upright: legs extended downward (negative z in body frame)
            upright_z = base_pos[2] + self.leg_retract_height * 0.4
            
            # Inverted: legs tucked upward relative to inverted body
            # When inverted, body frame z points down, so positive z brings legs toward body center
            inverted_z = base_pos[2] + self.inverted_tuck_offset['z']
            
            # Blend between upright and inverted configurations based on roll angle
            # When cos_roll = 1 (upright), use upright_z
            # When cos_roll = -1 (inverted), use inverted_z
            # The blend ensures smooth transition through 90-degree orientations
            blend_factor = 0.5 - 0.5 * cos_roll  # 0.0 upright, 1.0 inverted
            foot[2] = upright_z * (1.0 - blend_factor) + inverted_z * blend_factor
            
            # Lateral tucking with reduced amplitude during inversion
            lateral_amplitude = self.leg_lateral_swing * self.inverted_tuck_offset['lateral_factor']
            foot[1] = base_pos[1] + lateral_sign * lateral_amplitude * (0.5 + 0.5 * np.sin(np.pi * (t - 0.5)))
            
            # Minimal longitudinal adjustment
            long_offset = 0.02 * np.sin(np.pi * t)
            if is_front:
                foot[0] = base_pos[0] + long_offset
            else:
                foot[0] = base_pos[0] - long_offset
                
        elif phase < 0.78:
            # Recovery phase: transition from inverted-tucked to upright-extended
            t = (phase - 0.6) / 0.18
            smooth_t = 3 * t**2 - 2 * t**3
            
            # Current state based on roll (still transitioning out of inversion)
            blend_factor = 0.5 - 0.5 * cos_roll  # Current inversion state
            upright_z = base_pos[2] + self.leg_retract_height * 0.4 * (1.0 - smooth_t)
            inverted_z = base_pos[2] + self.inverted_tuck_offset['z'] * (1.0 - smooth_t)
            
            foot[2] = upright_z * (1.0 - blend_factor) + inverted_z * blend_factor
            
            # Return lateral position smoothly
            lateral_amplitude = self.leg_lateral_swing * self.inverted_tuck_offset['lateral_factor']
            lateral_amount = lateral_amplitude * (0.5 + 0.5 * np.sin(np.pi * (0.5 + 0.35 * (1.0 - smooth_t))))
            foot[1] = base_pos[1] + lateral_sign * lateral_amount * (1.0 - smooth_t)
            
            # Return longitudinal position
            long_offset = 0.02 * np.sin(np.pi * (1.0 - smooth_t))
            if is_front:
                foot[0] = base_pos[0] + long_offset
            else:
                foot[0] = base_pos[0] - long_offset
                
        else:
            # Landing phase: smooth extension to nominal stance (now fully upright)
            t = (phase - 0.78) / 0.22
            smooth_t = 3 * t**2 - 2 * t**3
            
            # Smooth downward extension for landing preparation
            foot[2] = base_pos[2] - 0.02 * smooth_t
            
            # Return to base lateral and longitudinal positions
            foot[1] = base_pos[1]
            foot[0] = base_pos[0]
        
        return foot