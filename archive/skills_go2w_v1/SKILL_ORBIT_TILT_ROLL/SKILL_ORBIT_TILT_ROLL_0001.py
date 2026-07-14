from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ORBIT_TILT_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Orbit with synchronized roll oscillation gait.

    The robot travels in a circular orbit while continuously oscillating its base roll angle.
    Roll motion is synchronized to orbital position:
    - Phase 0.0-0.25: Roll left (negative roll rate), reach max left at 0.25
    - Phase 0.25-0.5: Roll right (positive roll rate), reach max right at 0.5
    - Phase 0.5-0.75: Roll left (negative roll rate), reach max left at 0.75
    - Phase 0.75-1.0: Roll right (positive roll rate), return to neutral at 1.0

    All four legs remain in contact throughout. Outer legs (relative to roll direction)
    extend more, inner legs retract to support the tilting base.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        # Initialize base class
        super().__init__(initial_foot_positions_body, freq=1.0)

        self.leg_names = leg_names
        self.freq = 0.5  # Complete one orbit per 2 seconds at default time scale

        # Base foot positions (nominal stance in body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Orbital motion parameters
        self.vx = 0.4  # Forward velocity (m/s)
        self.yaw_rate = 0.8  # Constant yaw rate (rad/s) for circular motion
        # Orbit radius = vx / yaw_rate = 0.5 m

        # Roll oscillation parameters
        self.roll_rate_amplitude = 1.2  # Peak roll rate (rad/s)
        # Max roll angle = roll_rate_amplitude / (4 * freq) ~ 0.6 rad ~ 34 degrees

        # Leg extension modulation parameters
        self.lateral_extension_gain = 0.08  # Meters of lateral shift per radian of roll
        self.vertical_extension_gain = 0.04  # Meters of vertical shift per radian of roll

        # Track accumulated roll angle for leg extension computation
        self.current_roll_angle = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity, constant yaw rate,
        and phase-synchronized oscillating roll rate.
        """
        # Constant forward velocity and yaw rate for circular orbit
        vx = self.vx
        yaw_rate = self.yaw_rate

        # Roll rate oscillation synchronized to phase
        # Pattern: negative -> positive -> negative -> positive over [0,1]
        # Use sinusoidal modulation: roll_rate = amplitude * sin(4*pi*phase)
        # This gives: phase 0->0.25 negative, 0.25->0.5 positive, 0.5->0.75 negative, 0.75->1.0 positive
        roll_rate = self.roll_rate_amplitude * np.sin(4 * np.pi * phase)

        # Set velocity commands
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, yaw_rate])

        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

        # Update current roll angle for leg extension computation
        self.current_roll_angle += roll_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame.

        All feet remain in contact (stance). Leg extension is modulated based on
        current roll angle:
        - When rolling left (negative roll): right legs extend, left legs retract
        - When rolling right (positive roll): left legs extend, right legs retract

        Extension is both lateral (y-direction) and vertical (z-direction) to maintain
        contact and balance during roll oscillation.
        """
        # Start with nominal foot position
        foot = self.base_feet_pos_body[leg_name].copy()

        # Determine if this is a left or right leg
        is_left_leg = leg_name.startswith(('FL', 'RL'))

        # Compute lateral and vertical extension based on roll angle
        # Positive roll angle = rolled right, negative = rolled left

        if is_left_leg:
            # Left legs: extend when rolling right (positive roll), retract when rolling left
            lateral_shift = self.current_roll_angle * self.lateral_extension_gain
            vertical_shift = -abs(self.current_roll_angle) * self.vertical_extension_gain
        else:
            # Right legs: extend when rolling left (negative roll), retract when rolling right
            lateral_shift = -self.current_roll_angle * self.lateral_extension_gain
            vertical_shift = -abs(self.current_roll_angle) * self.vertical_extension_gain

        # Apply shifts
        # y-direction: lateral (positive = left in body frame)
        foot[1] += lateral_shift
        # z-direction: vertical (negative = down in body frame)
        foot[2] += vertical_shift

        return foot