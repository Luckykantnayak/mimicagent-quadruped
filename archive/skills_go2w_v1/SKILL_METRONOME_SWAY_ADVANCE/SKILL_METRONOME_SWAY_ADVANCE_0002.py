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
        self.leg_x_stride_offset = 0.08  # Apparent rearward motion during surge (m)
        
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
        # Using smoothed step functions for continuous velocity profile
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
        # Positive vy = right, matches rightward roll phase [0.0, 0.25]
        # Negative vy = left, matches leftward roll phase [0.5, 0.75]
        # Returns to center during transitions
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
        1. Roll-induced vertical geometry changes (z-axis) - computed to maintain world z-height
        2. Forward surge-induced apparent rearward motion (x-axis)
        
        The key insight: when the body rolls by angle φ, a foot at lateral offset y_body from
        centerline experiences a world-frame vertical displacement. To maintain constant world
        z-position (ground contact), the body-frame z-position must compensate.
        
        For a foot at lateral offset y_body under roll φ:
        - World z displacement ≈ y_body * sin(φ) for the roll rotation
        - Body-frame z adjustment needed ≈ -y_body * tan(φ) to counteract this
        
        Left legs (positive y_body): rolling right (positive roll) lifts them in world frame,
        so body-frame z must decrease (negative adjustment).
        
        Right legs (negative y_body): rolling right (positive roll) lowers them in world frame,
        so body-frame z must increase (positive adjustment).
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Current roll angle for this phase
        roll_angle = self.roll_amplitude * np.sin(2 * np.pi * phase)
        
        # Lateral offset of this foot from body centerline (y-coordinate in body frame)
        y_body = foot[1]
        
        # Compute body-frame z adjustment to maintain constant world z under roll
        # When body rolls by φ, a point at lateral offset y experiences world vertical change
        # To compensate, adjust body z by approximately -y * tan(φ)
        # For small to moderate angles, this maintains ground contact
        # For 30° roll: tan(30°) ≈ 0.577
        if abs(roll_angle) < 1e-6:
            z_adjustment = 0.0
        else:
            z_adjustment = -y_body * np.tan(roll_angle)
        
        foot[2] += z_adjustment
        
        # Fore-aft offset (x-axis in body frame) due to forward surge
        # During forward surge, feet appear to move rearward in body frame
        # Surge phases: ~0.25-0.5 and ~0.75-1.0
        # Use same surge activation as velocity computation
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