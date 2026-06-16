from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_METRONOME_SWAY_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Metronome sway advance gait with large-amplitude lateral base rolling.
    
    Motion characteristics:
    - Base rolls between ±30° in a pendulum-like oscillation
    - Forward surges occur during neutral roll transitions
    - All four feet maintain continuous ground contact
    - Legs compress/extend asymmetrically to accommodate lateral sway
    - Right legs compress during right sway, left legs compress during left sway
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for large amplitude sway motion
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.max_roll_angle = np.deg2rad(30)  # ±30° roll amplitude
        self.forward_surge_speed = 0.4  # Forward velocity during neutral transitions
        self.lateral_velocity_amp = 0.3  # Lateral velocity during sway phases
        self.roll_rate_amp = 1.2  # Roll rate amplitude (rad/s)
        
        # Leg motion parameters
        self.compression_amount = 0.06  # Vertical compression when weight-bearing
        self.extension_amount = 0.06  # Vertical extension when unloaded
        self.lateral_shift_amp = 0.04  # Lateral foot shift in body frame during roll
        self.rearward_drift = 0.03  # Rearward foot drift during forward surge
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with metronome-like lateral sway and forward surges.
        
        Phase structure:
        - [0.0, 0.25]: Right sway - positive roll rate, rightward lateral velocity
        - [0.25, 0.5]: Neutral surge 1 - negative roll rate (return to neutral), forward surge
        - [0.5, 0.75]: Left sway - negative roll rate, leftward lateral velocity
        - [0.75, 1.0]: Neutral surge 2 - positive roll rate (return to neutral), forward surge
        """
        
        # Determine current sub-phase
        if phase < 0.25:
            # Right sway phase
            sub_phase = phase / 0.25
            roll_rate = self.roll_rate_amp * np.sin(np.pi * sub_phase)
            vx = 0.05  # Minimal forward velocity during sway
            vy = self.lateral_velocity_amp * np.sin(np.pi * sub_phase)
            
        elif phase < 0.5:
            # Neutral surge 1 (right to neutral)
            sub_phase = (phase - 0.25) / 0.25
            roll_rate = -self.roll_rate_amp * np.sin(np.pi * sub_phase)
            vx = self.forward_surge_speed * np.sin(np.pi * sub_phase)
            # Lateral velocity transitions from right to left
            vy = self.lateral_velocity_amp * np.cos(np.pi * (sub_phase + 0.5))
            
        elif phase < 0.75:
            # Left sway phase
            sub_phase = (phase - 0.5) / 0.25
            roll_rate = -self.roll_rate_amp * np.sin(np.pi * sub_phase)
            vx = 0.05  # Minimal forward velocity during sway
            vy = -self.lateral_velocity_amp * np.sin(np.pi * sub_phase)
            
        else:
            # Neutral surge 2 (left to neutral)
            sub_phase = (phase - 0.75) / 0.25
            roll_rate = self.roll_rate_amp * np.sin(np.pi * sub_phase)
            vx = self.forward_surge_speed * np.sin(np.pi * sub_phase)
            # Lateral velocity transitions from left to right
            vy = -self.lateral_velocity_amp * np.cos(np.pi * (sub_phase + 0.5))
        
        # Set velocity commands (world frame)
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with compression/extension and lateral shifts.
        
        Right legs (FR, RR): compress during [0.0, 0.25], extend during [0.5, 0.75]
        Left legs (FL, RL): extend during [0.0, 0.25], compress during [0.5, 0.75]
        All legs equalize during neutral transitions [0.25, 0.5] and [0.75, 1.0]
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a right-side or left-side leg
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Vertical displacement (z-axis) and lateral shift (y-axis)
        z_offset = 0.0
        y_offset = 0.0
        x_offset = 0.0
        
        if phase < 0.25:
            # Right sway phase
            sub_phase = phase / 0.25
            if is_right_leg:
                # Right legs compress (move up in body frame)
                z_offset = self.compression_amount * np.sin(np.pi * sub_phase)
                y_offset = self.lateral_shift_amp * np.sin(np.pi * sub_phase)
            else:
                # Left legs extend (move down in body frame)
                z_offset = -self.extension_amount * np.sin(np.pi * sub_phase)
                y_offset = -self.lateral_shift_amp * np.sin(np.pi * sub_phase)
                
        elif phase < 0.5:
            # Neutral surge 1
            sub_phase = (phase - 0.25) / 0.25
            if is_right_leg:
                # Right legs transition from compressed to neutral
                z_offset = self.compression_amount * np.cos(np.pi * sub_phase)
                y_offset = self.lateral_shift_amp * np.cos(np.pi * sub_phase)
            else:
                # Left legs transition from extended to neutral
                z_offset = -self.extension_amount * np.cos(np.pi * sub_phase)
                y_offset = -self.lateral_shift_amp * np.cos(np.pi * sub_phase)
            # Rearward drift due to forward base motion
            x_offset = -self.rearward_drift * np.sin(np.pi * sub_phase)
            
        elif phase < 0.75:
            # Left sway phase
            sub_phase = (phase - 0.5) / 0.25
            if is_left_leg:
                # Left legs compress (move up in body frame)
                z_offset = self.compression_amount * np.sin(np.pi * sub_phase)
                y_offset = -self.lateral_shift_amp * np.sin(np.pi * sub_phase)
            else:
                # Right legs extend (move down in body frame)
                z_offset = -self.extension_amount * np.sin(np.pi * sub_phase)
                y_offset = self.lateral_shift_amp * np.sin(np.pi * sub_phase)
                
        else:
            # Neutral surge 2
            sub_phase = (phase - 0.75) / 0.25
            if is_left_leg:
                # Left legs transition from compressed to neutral
                z_offset = self.compression_amount * np.cos(np.pi * sub_phase)
                y_offset = -self.lateral_shift_amp * np.cos(np.pi * sub_phase)
            else:
                # Right legs transition from extended to neutral
                z_offset = -self.extension_amount * np.cos(np.pi * sub_phase)
                y_offset = self.lateral_shift_amp * np.cos(np.pi * sub_phase)
            # Rearward drift due to forward base motion
            x_offset = -self.rearward_drift * np.sin(np.pi * sub_phase)
        
        # Apply offsets
        foot[0] += x_offset
        foot[1] += y_offset
        foot[2] += z_offset
        
        return foot