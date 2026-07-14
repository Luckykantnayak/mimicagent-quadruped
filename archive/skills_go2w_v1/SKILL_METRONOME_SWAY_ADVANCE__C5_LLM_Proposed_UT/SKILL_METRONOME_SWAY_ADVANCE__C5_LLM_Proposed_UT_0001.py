from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_METRONOME_SWAY_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Metronome sway advance gait with large-amplitude lateral rolling (±30°).
    
    Motion characteristics:
    - Base rolls rhythmically left and right like an inverted pendulum metronome
    - Forward surges occur during neutral roll transitions
    - All four feet maintain continuous ground contact
    - Legs compress/extend asymmetrically to accommodate lateral weight shifts
    
    Phase structure:
    - [0.0, 0.25]: Right sway - roll right to +30°, right legs compress
    - [0.25, 0.5]: Neutral surge 1 - return to neutral, forward surge
    - [0.5, 0.75]: Left sway - roll left to -30°, left legs compress
    - [0.75, 1.0]: Neutral surge 2 - return to neutral, forward surge
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for large amplitude rolling motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.max_roll_angle = np.deg2rad(30)  # ±30° roll amplitude
        self.surge_velocity = 0.4  # Forward velocity during surge phases
        self.drift_velocity = 0.05  # Minimal forward drift during sway phases
        self.lateral_shift_velocity = 0.15  # Lateral velocity during weight shifts
        
        # Leg compression parameters
        self.compression_amount = 0.06  # Vertical compression for loaded legs
        self.extension_amount = 0.04  # Vertical extension for unloaded legs
        self.lateral_shift_amount = 0.02  # Lateral shift in body frame
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with metronome-like rolling and forward surges.
        
        Roll profile: sinusoidal modulation to achieve ±30° at phases 0.25 and 0.75
        Forward velocity: high during neutral transitions, low during sway phases
        Lateral velocity: coordinated with roll to create weight shift
        """
        
        # Compute target roll angle (sinusoidal profile)
        # Roll = max_roll * sin(2π * phase)
        # This gives: 0° at phase 0, +30° at 0.25, 0° at 0.5, -30° at 0.75, 0° at 1.0
        target_roll = self.max_roll_angle * np.sin(2 * np.pi * phase)
        
        # Roll rate is derivative of target roll
        # d(roll)/dt = max_roll * 2π * freq * cos(2π * phase)
        roll_rate = self.max_roll_angle * 2 * np.pi * self.freq * np.cos(2 * np.pi * phase)
        
        # Forward velocity: high during neutral transitions, low during sway
        # Use cosine squared to create peaks at 0.25-0.5 and 0.75-1.0
        if 0.0 <= phase < 0.25:
            # Right sway phase - minimal forward progress
            vx = self.drift_velocity
        elif 0.25 <= phase < 0.5:
            # Neutral surge 1 - forward surge
            surge_phase = (phase - 0.25) / 0.25
            vx = self.drift_velocity + self.surge_velocity * np.sin(np.pi * surge_phase)
        elif 0.5 <= phase < 0.75:
            # Left sway phase - minimal forward progress
            vx = self.drift_velocity
        else:
            # Neutral surge 2 - forward surge
            surge_phase = (phase - 0.75) / 0.25
            vx = self.drift_velocity + self.surge_velocity * np.sin(np.pi * surge_phase)
        
        # Lateral velocity: coordinated with roll direction
        # Positive vy (rightward) during right sway, negative (leftward) during left sway
        if 0.0 <= phase < 0.25:
            # Moving right during right sway
            lateral_progress = phase / 0.25
            vy = self.lateral_shift_velocity * np.sin(np.pi * lateral_progress)
        elif 0.25 <= phase < 0.5:
            # Returning to center from right
            lateral_progress = (phase - 0.25) / 0.25
            vy = -self.lateral_shift_velocity * np.sin(np.pi * lateral_progress)
        elif 0.5 <= phase < 0.75:
            # Moving left during left sway
            lateral_progress = (phase - 0.5) / 0.25
            vy = -self.lateral_shift_velocity * np.sin(np.pi * lateral_progress)
        else:
            # Returning to center from left
            lateral_progress = (phase - 0.75) / 0.25
            vy = self.lateral_shift_velocity * np.sin(np.pi * lateral_progress)
        
        # Vertical velocity: subtle compensation for geometric height changes
        # Base lowers slightly during sway (compression) and rises during neutral
        if 0.0 <= phase < 0.25 or 0.5 <= phase < 0.75:
            # Lowering during sway phases
            vz = -0.02
        else:
            # Rising during neutral phases
            vz = 0.02
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame with compression/extension pattern.
        
        Right legs (FR, RR): compress during right sway [0.0-0.25], extend during left sway [0.5-0.75]
        Left legs (FL, RL): extend during right sway [0.0-0.25], compress during left sway [0.5-0.75]
        All legs equalize during neutral transitions
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a right-side or left-side leg
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compute compression/extension based on phase and leg side
        if 0.0 <= phase < 0.25:
            # Right sway phase
            progress = phase / 0.25
            if is_right_leg:
                # Right legs compress (shorten)
                z_offset = -self.compression_amount * np.sin(np.pi * progress)
                y_offset = self.lateral_shift_amount * np.sin(np.pi * progress)
            else:
                # Left legs extend (lengthen)
                z_offset = self.extension_amount * np.sin(np.pi * progress)
                y_offset = -self.lateral_shift_amount * np.sin(np.pi * progress)
        
        elif 0.25 <= phase < 0.5:
            # Neutral surge 1 - transition to neutral
            progress = (phase - 0.25) / 0.25
            if is_right_leg:
                # Right legs extending back to neutral
                z_offset = -self.compression_amount * np.sin(np.pi * (1 - progress))
                y_offset = self.lateral_shift_amount * np.sin(np.pi * (1 - progress))
            else:
                # Left legs compressing back to neutral
                z_offset = self.extension_amount * np.sin(np.pi * (1 - progress))
                y_offset = -self.lateral_shift_amount * np.sin(np.pi * (1 - progress))
        
        elif 0.5 <= phase < 0.75:
            # Left sway phase
            progress = (phase - 0.5) / 0.25
            if is_left_leg:
                # Left legs compress (shorten)
                z_offset = -self.compression_amount * np.sin(np.pi * progress)
                y_offset = -self.lateral_shift_amount * np.sin(np.pi * progress)
            else:
                # Right legs extend (lengthen)
                z_offset = self.extension_amount * np.sin(np.pi * progress)
                y_offset = self.lateral_shift_amount * np.sin(np.pi * progress)
        
        else:
            # Neutral surge 2 - transition to neutral
            progress = (phase - 0.75) / 0.25
            if is_left_leg:
                # Left legs extending back to neutral
                z_offset = -self.compression_amount * np.sin(np.pi * (1 - progress))
                y_offset = -self.lateral_shift_amount * np.sin(np.pi * (1 - progress))
            else:
                # Right legs compressing back to neutral
                z_offset = self.extension_amount * np.sin(np.pi * (1 - progress))
                y_offset = self.lateral_shift_amount * np.sin(np.pi * (1 - progress))
        
        # Apply offsets
        foot[1] += y_offset  # Lateral shift in body frame
        foot[2] += z_offset  # Vertical compression/extension
        
        return foot