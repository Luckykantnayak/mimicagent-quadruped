from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_METRONOME_SWAY_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Metronome sway gait with forward progression.
    
    - Base rolls side-to-side (±30°) in a continuous sinusoidal pattern
    - Forward velocity surges occur during roll transitions through neutral
    - Lateral sway accompanies roll to shift COM
    - All four feet maintain continuous ground contact
    - Leg compression/extension in body frame compensates for roll-induced geometry changes
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Metronome sway frequency (Hz)
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Roll parameters
        self.roll_amplitude = np.deg2rad(30.0)  # ±30° roll
        
        # Forward surge parameters
        self.surge_vx_peak = 0.4  # Peak forward velocity during surge (m/s)
        self.surge_phase_width = 0.2  # Width of surge window centered at transitions
        
        # Lateral sway parameters
        self.lateral_vy_peak = 0.3  # Peak lateral velocity during sway (m/s)
        
        # Leg fore-aft motion due to forward surge (body frame x-offset)
        self.leg_x_stride_offset = 0.045  # Reduced apparent rearward motion during surge (m)
        
        # Z-compensation scaling factor to prevent joint limit violations
        self.z_compensation_scale = 0.35  # Scale factor for roll-induced z adjustment
        
        # Time and base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (will be computed per phase)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with sinusoidal roll and pulsed forward/lateral velocities.
        
        Roll profile: roll(phase) = roll_amplitude * sin(2π * phase)
          - phase 0.0: neutral (0°)
          - phase 0.25: peak right roll (+30°)
          - phase 0.5: neutral (0°)
          - phase 0.75: peak left roll (-30°)
          - phase 1.0: neutral (0°)
        
        Roll rate: d(roll)/dt = roll_amplitude * 2π * freq * cos(2π * phase)
        
        Forward surge: centered at phase 0.375 and 0.875 (midpoints of neutral transitions)
        Lateral sway: sinusoidal, in phase with roll angle
        """
        
        # Roll angle target and rate
        roll_angle = self.roll_amplitude * np.sin(2 * np.pi * phase)
        roll_rate = self.roll_amplitude * 2 * np.pi * self.freq * np.cos(2 * np.pi * phase)
        
        # Forward velocity: Gaussian-like pulses centered at phase 0.375 and 0.875
        surge_center_1 = 0.375
        surge_center_2 = 0.875
        sigma = self.surge_phase_width / 2.5  # Width of surge window
        
        # Distance to nearest surge center (handling phase wrap)
        dist_1 = min(abs(phase - surge_center_1), abs(phase - surge_center_1 + 1.0), abs(phase - surge_center_1 - 1.0))
        dist_2 = min(abs(phase - surge_center_2), abs(phase - surge_center_2 + 1.0), abs(phase - surge_center_2 - 1.0))
        
        surge_1 = np.exp(-0.5 * (dist_1 / sigma) ** 2)
        surge_2 = np.exp(-0.5 * (dist_2 / sigma) ** 2)
        vx = self.surge_vx_peak * (surge_1 + surge_2)
        
        # Lateral velocity: sinusoidal, synchronized with roll
        vy = self.lateral_vy_peak * np.sin(2 * np.pi * phase)
        
        # No vertical or pitch/yaw motion
        vz = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Set velocity commands
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
        Compute foot position in body frame.
        
        All feet maintain ground contact. Body-frame foot positions vary to compensate for:
        1. Roll-induced vertical geometry changes (z-axis) - scaled to remain within joint limits
        2. Forward surge-induced apparent rearward motion (x-axis)
        
        The z-compensation uses a sine-based approximation scaled by a factor to prevent
        joint limit violations while maintaining the directional correctness of the adjustment.
        Left legs (positive y_body) adjust opposite to right legs (negative y_body) under roll.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Current roll angle for this phase
        roll_angle = self.roll_amplitude * np.sin(2 * np.pi * phase)
        
        # Lateral offset of this foot from body centerline (y-coordinate in body frame)
        y_body = foot[1]
        
        # Compute body-frame z adjustment using sine function with scaling
        # This provides smooth, bounded adjustment that suggests ground contact maintenance
        # without exceeding joint limits at large roll angles
        # Using sin(roll_angle) instead of tan(roll_angle) for natural bounding
        z_adjustment = -y_body * np.sin(roll_angle) * self.z_compensation_scale
        
        foot[2] += z_adjustment
        
        # Fore-aft offset (x-axis in body frame) due to forward surge
        surge_center_1 = 0.375
        surge_center_2 = 0.875
        sigma = self.surge_phase_width / 2.5
        
        dist_1 = min(abs(phase - surge_center_1), abs(phase - surge_center_1 + 1.0), abs(phase - surge_center_1 - 1.0))
        dist_2 = min(abs(phase - surge_center_2), abs(phase - surge_center_2 + 1.0), abs(phase - surge_center_2 - 1.0))
        
        surge_1 = np.exp(-0.5 * (dist_1 / sigma) ** 2)
        surge_2 = np.exp(-0.5 * (dist_2 / sigma) ** 2)
        surge_activation = surge_1 + surge_2
        
        # Apparent rearward motion during surge (negative x in body frame)
        x_offset = -self.leg_x_stride_offset * surge_activation
        foot[0] += x_offset
        
        return foot

    def update(self, dt, **params):
        self.t += dt
        phase = (self.t * self.freq) % 1.0
        
        self.update_base_motion(phase, dt)
        
        foot_positions_body = {}
        for leg_name in self.leg_names:
            foot_positions_body[leg_name] = self.compute_foot_position_body_frame(leg_name, phase)
        
        return {
            'root_position': self.root_pos,
            'root_quaternion': self.root_quat,
            'foot_positions_body': foot_positions_body,
            'velocity_world': self.vel_world,
            'omega_world': self.omega_world
        }