from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_METRONOME_SWAY_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Metronome sway advance gait with large-amplitude lateral base rolling (±30°).
    
    Motion characteristics:
    - All four feet maintain ground contact throughout the cycle
    - Base rolls laterally between +30° (right) and -30° (left)
    - Forward surges occur during neutral roll transitions
    - Legs compress/extend asymmetrically to accommodate lateral sway
    - Right legs compress during right sway, left legs compress during left sway
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for large amplitude sway motion
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - reduced roll angle for initial stability
        self.max_roll_angle = np.deg2rad(20.0)  # Reduced from 30° to 20° for joint safety
        self.leg_compression_range = 0.26  # Increased from 0.18 to 0.26 for more vertical travel
        
        # Forward surge parameters
        self.surge_velocity = 0.8  # Forward velocity during neutral phases
        self.drift_velocity = 0.1  # Minimal forward drift during sway phases
        
        # Lateral motion parameters
        self.lateral_velocity_amp = 0.3  # Lateral velocity amplitude
        
        # Vertical motion parameters - reduced to minimize base sinking during compression
        self.vertical_velocity_amp = 0.06  # Reduced from 0.15 to 0.06
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with lateral rolling and forward surges.
        
        Phase structure:
        - [0.0, 0.25]: Right sway - roll right, lateral right shift, minimal forward
        - [0.25, 0.5]: Neutral transition 1 - unroll to neutral, forward surge
        - [0.5, 0.75]: Left sway - roll left, lateral left shift, minimal forward
        - [0.75, 1.0]: Neutral transition 2 - unroll to neutral, forward surge
        """
        
        # Compute target roll angle using smooth sinusoidal profile
        # Peak right (+20°) at phase 0.25, peak left (-20°) at phase 0.75
        target_roll = self.max_roll_angle * np.sin(2 * np.pi * phase)
        
        # Compute roll rate to achieve target roll angle
        roll_rate = self.max_roll_angle * 2 * np.pi * self.freq * np.cos(2 * np.pi * phase)
        
        # Forward velocity: high during neutral transitions, low during sway
        if 0.25 <= phase < 0.5 or 0.75 <= phase < 1.0:
            # Neutral transition phases - forward surge
            vx = self.surge_velocity
        else:
            # Sway phases - minimal forward drift
            vx = self.drift_velocity
        
        # Lateral velocity: rightward during right sway, leftward during left sway
        vy = self.lateral_velocity_amp * np.cos(2 * np.pi * phase)
        
        # Vertical velocity: rise during peak compression to reduce joint stress
        # Inverted pattern - base rises when legs compress (phase 0.25, 0.75)
        vz = self.vertical_velocity_amp * np.sin(2 * np.pi * phase)
        
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
        Enhanced with roll-dependent geometric corrections.
        
        Right legs (FR, RR):
        - Compress during right sway [0.0, 0.25]
        - Extend to neutral during transition [0.25, 0.5]
        - Extend during left sway [0.5, 0.75]
        - Return to neutral during transition [0.75, 1.0]
        
        Left legs (FL, RL):
        - Extend during right sway [0.0, 0.25]
        - Return to neutral during transition [0.25, 0.5]
        - Compress during left sway [0.5, 0.75]
        - Extend to neutral during transition [0.75, 1.0]
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a right-side or left-side leg
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Compute current roll angle for geometric correction
        current_roll = self.max_roll_angle * np.sin(2 * np.pi * phase)
        
        # Base vertical offset pattern with phase shift to avoid peak stress
        # Phase shift of 0.08 means compression peaks slightly before roll peaks
        phase_shifted = phase + 0.08
        if phase_shifted > 1.0:
            phase_shifted -= 1.0
        
        # Compute vertical offset based on phase and leg side
        if is_right_leg:
            # Right legs: compress when phase near 0.25, extend when phase near 0.75
            vertical_offset = -self.leg_compression_range * 0.5 * np.sin(2 * np.pi * phase_shifted)
        else:
            # Left legs: extend when phase near 0.25, compress when phase near 0.75
            vertical_offset = self.leg_compression_range * 0.5 * np.sin(2 * np.pi * phase_shifted)
        
        # Add roll-dependent geometric correction for vertical displacement
        # When base rolls, feet at lateral offset need additional vertical adjustment
        lateral_offset = abs(foot[1])  # Distance from body centerline
        roll_vertical_correction = lateral_offset * abs(np.sin(current_roll))
        
        if is_right_leg:
            # Right legs need more compression when rolling right (positive roll)
            if current_roll > 0:
                vertical_offset -= roll_vertical_correction * 0.5
        else:
            # Left legs need more compression when rolling left (negative roll)
            if current_roll < 0:
                vertical_offset -= roll_vertical_correction * 0.5
        
        # Apply vertical offset
        foot[2] += vertical_offset
        
        # Enhanced lateral adjustment to account for roll geometry
        # Compressed side legs move inward, extended side legs move outward
        lateral_adjustment_base = 0.10  # Increased from 0.05 to 0.10
        lateral_adjustment = lateral_adjustment_base * np.sin(2 * np.pi * phase)
        
        # Add roll-dependent lateral correction
        # When rolling, compressed side needs to move inward more
        roll_lateral_correction = lateral_offset * (1.0 - np.cos(abs(current_roll)))
        
        if is_right_leg:
            foot[1] -= lateral_adjustment  # Move inward during right roll
            if current_roll > 0:
                foot[1] -= roll_lateral_correction * 0.3
        else:
            foot[1] += lateral_adjustment  # Move outward during right roll
            if current_roll < 0:
                foot[1] += roll_lateral_correction * 0.3
        
        return foot