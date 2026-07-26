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
        
        # Base foot positions in body frame (with increased clearance)
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            pos = v.copy()
            # Reduce leg extension (make z less negative) to increase base height clearance
            pos[2] += 0.08  # Retract feet 8cm to lift base higher
            self.base_feet_pos_body[k] = pos
        
        # Wobble parameters - reduced amplitude for safer motion
        self.pitch_amplitude = np.deg2rad(10.0)  # ±10° pitch oscillation (reduced from 12°)
        self.roll_amplitude = np.deg2rad(10.0)   # ±10° roll oscillation (reduced from 12°)
        self.yaw_rate = 2 * np.pi * self.freq    # 360° per cycle (rad/s)
        
        # Wobble frequency: 2 complete oscillations per phase cycle
        self.wobble_freq = 2.0
        
        # Phase offset between pitch and roll (0.25 creates 90° phase shift)
        self.pitch_roll_phase_offset = 0.25
        
        # Leg extension adjustments for wobble compensation - reduced magnitudes
        self.front_extension_scale = 0.04  # Forward/backward adjustment range (reduced)
        self.rear_extension_scale = 0.04
        self.lateral_adjustment_scale = 0.03  # Left/right adjustment range (reduced)
        self.vertical_adjustment_scale = 0.025  # Vertical compensation range (reduced and will be inverted)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, 0.0])  # Will be set properly on first update
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
        
        CORRECTED APPROACH:
        - When base tilts, feet must retract toward body (less negative z) to maintain ground contact
        - Front legs adjust forward/back based on pitch
        - Left/right legs adjust laterally based on roll
        - Vertical retraction (positive adjustment to z) compensates for tilt magnitude
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute current pitch and roll angles from phase
        pitch_angle = self.pitch_amplitude * np.sin(2 * np.pi * self.wobble_freq * phase)
        roll_angle = self.roll_amplitude * np.sin(2 * np.pi * self.wobble_freq * (phase + self.pitch_roll_phase_offset))
        
        # Determine leg position (front/rear, left/right)
        is_front = leg_name.startswith('F')
        is_left = leg_name.endswith('L')
        
        # Pitch compensation: front legs extend forward with forward pitch, rear legs retract
        if is_front:
            # Front legs: positive pitch (nose down) → shift forward in body frame
            foot[0] += self.front_extension_scale * np.sin(pitch_angle)
        else:
            # Rear legs: positive pitch (nose down) → shift backward in body frame
            foot[0] -= self.rear_extension_scale * np.sin(pitch_angle)
        
        # Roll compensation: left legs adjust left with left roll, right legs adjust right
        if is_left:
            # Left legs: negative roll (left side down) → shift left
            foot[1] -= self.lateral_adjustment_scale * np.sin(roll_angle)
        else:
            # Right legs: positive roll (right side down) → shift right
            foot[1] += self.lateral_adjustment_scale * np.sin(roll_angle)
        
        # CORRECTED: Vertical compensation - retract legs (ADD to z, making it less negative)
        # When base tilts, feet must pull toward body to maintain constant world-frame ground height
        attitude_magnitude = np.sqrt(pitch_angle**2 + roll_angle**2)
        foot[2] += self.vertical_adjustment_scale * attitude_magnitude  # INVERTED: now adds instead of subtracts
        
        return foot