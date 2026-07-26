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
        self.pitch_amplitude = np.deg2rad(12.0)  # ±12° pitch oscillation
        self.roll_amplitude = np.deg2rad(12.0)   # ±12° roll oscillation
        self.yaw_rate = 2 * np.pi * self.freq    # 360° per cycle (rad/s)
        
        # Wobble frequency: 2 complete oscillations per phase cycle
        self.wobble_freq = 2.0
        
        # Phase offset between pitch and roll (0.25 creates 90° phase shift)
        self.pitch_roll_phase_offset = 0.25
        
        # Leg extension adjustments for wobble compensation
        self.front_extension_scale = 0.05  # Forward/backward adjustment range
        self.rear_extension_scale = 0.05
        self.lateral_adjustment_scale = 0.035  # Left/right adjustment range
        self.vertical_adjustment_scale = 0.018  # Vertical compensation range
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, 0.0])
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
        - Horizontal adjustments (x, y) shift feet to track ground as body rotates
        - Vertical adjustments (z) are asymmetric: legs on downward tilt side extend down,
          legs on upward tilt side retract up
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
        
        # Asymmetric vertical compensation based on which side of tilt the leg is on
        # Legs on downward side extend (negative z adjustment), upward side retract (positive z adjustment)
        
        # Pitch contribution to vertical adjustment:
        # Forward pitch (positive) tilts nose down, so front legs are on downward side (extend)
        # and rear legs are on upward side (retract)
        if is_front:
            pitch_vertical_contribution = -np.sin(pitch_angle)  # Negative = extend down when pitching forward
        else:
            pitch_vertical_contribution = +np.sin(pitch_angle)  # Positive = retract up when pitching forward
        
        # Roll contribution to vertical adjustment:
        # Right roll (positive) tilts right side down, so right legs are on downward side (extend)
        # and left legs are on upward side (retract)
        if is_left:
            roll_vertical_contribution = +np.sin(roll_angle)  # Positive = retract up when rolling right
        else:
            roll_vertical_contribution = -np.sin(roll_angle)  # Negative = extend down when rolling right
        
        # Combine pitch and roll contributions
        total_vertical_adjustment = self.vertical_adjustment_scale * (pitch_vertical_contribution + roll_vertical_contribution)
        foot[2] += total_vertical_adjustment
        
        return foot