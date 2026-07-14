from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_WOBBLE_PLATE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Wobble Plate Spin: Continuous 360° yaw rotation with coupled pitch-roll wobble.
    
    - Base rotates continuously at constant yaw rate (360° per cycle)
    - Pitch oscillates sinusoidally at 2x frequency (two full cycles per phase)
    - Roll oscillates sinusoidally at 2x frequency, 90° out of phase with pitch
    - All four legs maintain continuous ground contact throughout motion
    - Feet adjust dynamically in body frame to maintain ground contact as base wobbles
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Full cycle duration (Hz)
        
        # Base foot positions in body frame - use original positions without modification
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Wobble parameters
        self.pitch_amplitude = np.deg2rad(10.0)  # ±10° pitch oscillation
        self.roll_amplitude = np.deg2rad(10.0)   # ±10° roll oscillation
        self.yaw_rate = 2 * np.pi * self.freq    # 360° per cycle (rad/s)
        
        # Wobble frequency: 2 complete oscillations per phase cycle
        self.wobble_freq = 2.0
        
        # Phase offset between pitch and roll (0.25 creates 90° phase shift)
        self.pitch_roll_phase_offset = 0.25
        
        # Leg extension adjustments for wobble compensation
        self.front_extension_scale = 0.04  # Forward/backward adjustment range
        self.rear_extension_scale = 0.04
        self.lateral_adjustment_scale = 0.03  # Left/right adjustment range
        
        # Leg-specific vertical compensation scales (tuned per leg geometry)
        # RR is already safe at lower scale, FL/RL need more, FR intermediate
        self.vertical_scales = {
            'FL': 0.115,  # Front-left needs most compensation
            'FR': 0.095,  # Front-right moderate
            'RL': 0.105,  # Rear-left significant
            'RR': 0.075   # Rear-right already safe, keep conservative
        }
        
        # Base state - increased height for clearance
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, 0.038])  # Increased from 0.02 to 0.038
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (world frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base angular velocities to create wobble + spin motion.
        
        - Constant yaw rate for continuous 360° rotation
        - Sinusoidal pitch rate (2 cycles per phase)
        - Sinusoidal roll rate (2 cycles per phase, 90° out of phase with pitch)
        """
        # Linear velocity: zero (in-place motion)
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        # Constant yaw rate for continuous rotation
        yaw_rate = self.yaw_rate
        
        # Pitch and roll rates derived from sinusoidal oscillations
        wobble_angular_freq = 2 * np.pi * self.wobble_freq * self.freq
        
        # Pitch rate: derivative of pitch oscillation
        pitch_phase_arg = 2 * np.pi * self.wobble_freq * phase
        pitch_rate = self.pitch_amplitude * wobble_angular_freq * np.cos(pitch_phase_arg)
        
        # Roll rate: derivative of roll oscillation (with 90° phase offset)
        roll_phase_arg = 2 * np.pi * self.wobble_freq * (phase + self.pitch_roll_phase_offset)
        roll_rate = self.roll_amplitude * wobble_angular_freq * np.cos(roll_phase_arg)
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
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
        Compute foot position in body frame to maintain ground contact during wobble.
        
        Strategy:
        - Legs on downward tilt side RETRACT (positive z adjustment) because base tilt
          brings attachment point closer to ground
        - Legs on upward tilt side EXTEND (negative z adjustment)
        - Leg-specific scaling accounts for geometric differences
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute current pitch and roll angles from phase
        pitch_angle = self.pitch_amplitude * np.sin(2 * np.pi * self.wobble_freq * phase)
        roll_angle = self.roll_amplitude * np.sin(2 * np.pi * self.wobble_freq * (phase + self.pitch_roll_phase_offset))
        
        # Determine leg position (front/rear, left/right)
        is_front = leg_name.startswith('F')
        is_left = leg_name.endswith('L')
        
        # Pitch compensation: front legs shift forward with forward pitch, rear legs shift backward
        if is_front:
            foot[0] += self.front_extension_scale * np.sin(pitch_angle)
        else:
            foot[0] -= self.rear_extension_scale * np.sin(pitch_angle)
        
        # Roll compensation: left legs shift left with left roll, right legs shift right
        if is_left:
            foot[1] -= self.lateral_adjustment_scale * np.sin(roll_angle)
        else:
            foot[1] += self.lateral_adjustment_scale * np.sin(roll_angle)
        
        # Vertical compensation with corrected signs and leg-specific scaling
        # Pitch contribution:
        # - Forward pitch (positive) → front legs on downward side → RETRACT (positive)
        # - Forward pitch (positive) → rear legs on upward side → EXTEND (negative)
        if is_front:
            pitch_vertical_contribution = +np.sin(pitch_angle)
        else:
            pitch_vertical_contribution = -np.sin(pitch_angle)
        
        # Roll contribution:
        # - Right roll (positive) → right legs on downward side → RETRACT (positive)
        # - Right roll (positive) → left legs on upward side → EXTEND (negative)
        if is_left:
            roll_vertical_contribution = -np.sin(roll_angle)
        else:
            roll_vertical_contribution = +np.sin(roll_angle)
        
        # Apply leg-specific vertical scale
        vertical_scale = self.vertical_scales[leg_name]
        total_vertical_adjustment = vertical_scale * (pitch_vertical_contribution + roll_vertical_contribution)
        foot[2] += total_vertical_adjustment
        
        # Coupled horizontal-vertical adjustment: when feet shift horizontally to track ground,
        # add small vertical extension to follow arc trajectory
        horizontal_shift_x = self.front_extension_scale * np.sin(pitch_angle) if is_front else -self.rear_extension_scale * np.sin(pitch_angle)
        horizontal_shift_y = -self.lateral_adjustment_scale * np.sin(roll_angle) if is_left else self.lateral_adjustment_scale * np.sin(roll_angle)
        horizontal_displacement = np.sqrt(horizontal_shift_x**2 + horizontal_shift_y**2)
        foot[2] -= 0.018 * horizontal_displacement  # Slight extension for horizontal shifts
        
        return foot