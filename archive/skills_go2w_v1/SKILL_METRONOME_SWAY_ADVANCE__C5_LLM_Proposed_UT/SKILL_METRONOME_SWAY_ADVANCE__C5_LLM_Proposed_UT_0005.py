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
        self.compression_amount = 0.035  # Vertical compression for loaded legs
        self.extension_amount = 0.025  # Vertical extension for unloaded legs
        self.lateral_shift_amount = 0.02  # Lateral shift in body frame
        
        # Ground clearance safety margin - applied selectively to compressed legs
        self.ground_clearance_margin = 0.035  # Increased from 0.03m to 0.035m
        
        # Base state - increased slightly from 0.35m to 0.37m for startup buffer
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, 0.37])  # Optimized height for startup transient
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
        Vertical velocity: RISES during sway (compression) to maintain ground clearance
        """
        
        # Compute target roll angle (sinusoidal profile)
        target_roll = self.max_roll_angle * np.sin(2 * np.pi * phase)
        
        # Roll rate is derivative of target roll
        roll_rate = self.max_roll_angle * 2 * np.pi * self.freq * np.cos(2 * np.pi * phase)
        
        # Forward velocity: high during neutral transitions, low during sway
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
        
        # Vertical velocity: rise during sway to compensate for leg compression
        # Phase-advanced profile with initial boost for startup transient
        if 0.0 <= phase < 0.25:
            # Right sway - right legs compressing, base should rise
            sway_progress = phase / 0.25
            # Phase-advanced sine with startup boost
            vz_base = 0.12 * np.sin(np.pi * min(sway_progress + 0.15, 1.0))
            # Add initial boost for first 5% of phase to handle startup transient
            if phase < 0.05:
                vz = vz_base + 0.03
            else:
                vz = vz_base
        elif 0.25 <= phase < 0.5:
            # Neutral surge 1 - settling back down gently
            surge_progress = (phase - 0.25) / 0.25
            vz = -0.05 * np.sin(np.pi * surge_progress)
        elif 0.5 <= phase < 0.75:
            # Left sway - left legs compressing, base should rise
            sway_progress = (phase - 0.5) / 0.25
            # Phase-advanced sine (no startup boost needed, base already at stable height)
            vz = 0.12 * np.sin(np.pi * min(sway_progress + 0.15, 1.0))
        else:
            # Neutral surge 2 - settling back down gently
            surge_progress = (phase - 0.75) / 0.25
            vz = -0.05 * np.sin(np.pi * surge_progress)
        
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
        
        Negative z in body frame means foot is further below base (more extended leg).
        Ground clearance margin is applied SELECTIVELY only to compressed legs with phase-dependent scaling.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a right-side or left-side leg
        is_right_leg = leg_name in ['FR', 'RR']
        is_left_leg = leg_name in ['FL', 'RL']
        
        # Track whether this leg is compressed and the compression magnitude
        is_compressed = False
        smooth_factor = 0.0
        
        # Compute compression/extension based on phase and leg side
        if 0.0 <= phase < 0.25:
            # Right sway phase
            progress = phase / 0.25
            smooth_factor = np.sin(np.pi * progress)
            if is_right_leg:
                # Right legs compress (z becomes less negative = shorter leg)
                z_offset = self.compression_amount * smooth_factor
                y_offset = self.lateral_shift_amount * smooth_factor
                is_compressed = True
            else:
                # Left legs extend (z becomes more negative = longer leg)
                z_offset = -self.extension_amount * smooth_factor
                y_offset = -self.lateral_shift_amount * smooth_factor
        
        elif 0.25 <= phase < 0.5:
            # Neutral surge 1 - transition to neutral
            progress = (phase - 0.25) / 0.25
            smooth_factor = np.sin(np.pi * (1 - progress))
            if is_right_leg:
                # Right legs extending back to neutral
                z_offset = self.compression_amount * smooth_factor
                y_offset = self.lateral_shift_amount * smooth_factor
                is_compressed = (smooth_factor > 0.3)  # Still compressed if factor significant
            else:
                # Left legs compressing back to neutral
                z_offset = -self.extension_amount * smooth_factor
                y_offset = -self.lateral_shift_amount * smooth_factor
        
        elif 0.5 <= phase < 0.75:
            # Left sway phase
            progress = (phase - 0.5) / 0.25
            smooth_factor = np.sin(np.pi * progress)
            if is_left_leg:
                # Left legs compress (z becomes less negative = shorter leg)
                z_offset = self.compression_amount * smooth_factor
                y_offset = -self.lateral_shift_amount * smooth_factor
                is_compressed = True
            else:
                # Right legs extend (z becomes more negative = longer leg)
                z_offset = -self.extension_amount * smooth_factor
                y_offset = self.lateral_shift_amount * smooth_factor
        
        else:
            # Neutral surge 2 - transition to neutral
            progress = (phase - 0.75) / 0.25
            smooth_factor = np.sin(np.pi * (1 - progress))
            if is_left_leg:
                # Left legs extending back to neutral
                z_offset = self.compression_amount * smooth_factor
                y_offset = -self.lateral_shift_amount * smooth_factor
                is_compressed = (smooth_factor > 0.3)  # Still compressed if factor significant
            else:
                # Right legs compressing back to neutral
                z_offset = -self.extension_amount * smooth_factor
                y_offset = self.lateral_shift_amount * smooth_factor
        
        # Apply offsets
        foot[1] += y_offset  # Lateral shift in body frame
        foot[2] += z_offset  # Vertical compression/extension
        
        # Apply ground clearance safety margin SELECTIVELY with phase-dependent scaling
        # Only apply to compressed legs to avoid over-extending already-extended legs
        if is_compressed:
            # Scale margin by compression magnitude: full margin at peak, half at onset/offset
            margin_scale = max(smooth_factor, 0.5)
            effective_margin = self.ground_clearance_margin * margin_scale
            foot[2] -= effective_margin
        
        return foot