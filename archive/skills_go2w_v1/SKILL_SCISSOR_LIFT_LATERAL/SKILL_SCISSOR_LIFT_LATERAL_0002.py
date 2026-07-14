from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SCISSOR_LIFT_LATERAL_MotionGenerator(BaseMotionGenerator):
    """
    Scissor Lift Lateral Locomotion Skill.

    Achieves rightward (lateral +y) translation through alternating vertical
    leg extensions that create controlled tilting moments. All four feet
    remain in ground contact throughout the cycle.

    Phase structure:
      [0.0, 0.3]: Left legs extend, right legs compress → rightward tilt
      [0.3, 0.5]: Equalization → level base, maintain lateral momentum
      [0.5, 0.8]: Right legs extend, left legs compress → leftward tilt
      [0.8, 1.0]: Final equalization → return to neutral

    Leg motion maintains ground contact while modulating effective leg length.
    Base motion prescribes lateral velocity and roll oscillations consistent with geometry.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)

        # Vertical extension parameters
        self.extension_amplitude = 0.04  # Max vertical extension differential (m)
        
        # Lateral velocity parameters
        self.lateral_velocity_max = 0.12  # Max rightward velocity (m/s)

        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Estimate lateral spacing for roll calculation
        left_y = np.mean([initial_foot_positions_body[leg][1] for leg in leg_names if leg.startswith('FL') or leg.startswith('RL')])
        right_y = np.mean([initial_foot_positions_body[leg][1] for leg in leg_names if leg.startswith('FR') or leg.startswith('RR')])
        self.lateral_spacing = abs(left_y - right_y)

        # Identify leg groups by side
        self.left_legs = [leg for leg in leg_names if leg.startswith('FL') or leg.startswith('RL')]
        self.right_legs = [leg for leg in leg_names if leg.startswith('FR') or leg.startswith('RR')]

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def compute_differential_extension(self, phase):
        """
        Compute the differential extension between left and right legs.
        Positive means left legs extended relative to right legs.
        Returns (differential, left_extension, right_extension) where extensions are absolute.
        """
        if phase < 0.3:
            # Left legs extend, right legs compress
            progress = phase / 0.3
            # Smooth extension using cosine envelope
            envelope = (1.0 - np.cos(np.pi * progress)) / 2.0
            left_ext = self.extension_amplitude * envelope
            right_ext = 0.0
            differential = left_ext - right_ext
        elif phase < 0.5:
            # Equalization: smoothly return to neutral
            progress = (phase - 0.3) / 0.2
            envelope = (1.0 + np.cos(np.pi * progress)) / 2.0
            left_ext = self.extension_amplitude * envelope
            right_ext = 0.0
            differential = left_ext - right_ext
        elif phase < 0.8:
            # Right legs extend, left legs compress
            progress = (phase - 0.5) / 0.3
            envelope = (1.0 - np.cos(np.pi * progress)) / 2.0
            right_ext = self.extension_amplitude * envelope
            left_ext = 0.0
            differential = left_ext - right_ext
        else:
            # Final equalization
            progress = (phase - 0.8) / 0.2
            envelope = (1.0 + np.cos(np.pi * progress)) / 2.0
            right_ext = self.extension_amplitude * envelope
            left_ext = 0.0
            differential = left_ext - right_ext
        
        return differential, left_ext, right_ext

    def update_base_motion(self, phase, dt):
        """
        Update base motion with lateral velocity and roll oscillations
        consistent with differential leg extensions.
        """
        differential, left_ext, right_ext = self.compute_differential_extension(phase)
        
        # Compute target roll angle based on geometric differential
        if self.lateral_spacing > 0.01:
            target_roll = np.arctan2(differential, self.lateral_spacing)
        else:
            target_roll = 0.0
        
        # Limit roll angle to reasonable range
        target_roll = np.clip(target_roll, -0.15, 0.15)
        
        # Compute roll rate to approach target (derivative-based)
        if phase < 0.3:
            progress = phase / 0.3
            roll_rate = -0.3 * np.sin(np.pi * progress)
        elif phase < 0.5:
            progress = (phase - 0.3) / 0.2
            roll_rate = 0.4 * np.sin(np.pi * progress)
        elif phase < 0.8:
            progress = (phase - 0.5) / 0.3
            roll_rate = 0.3 * np.sin(np.pi * progress)
        else:
            progress = (phase - 0.8) / 0.2
            roll_rate = -0.4 * np.sin(np.pi * progress)
        
        # Lateral velocity profile
        if phase < 0.3:
            progress = phase / 0.3
            vy = self.lateral_velocity_max * (0.5 + 0.5 * (1.0 - np.cos(np.pi * progress)) / 2.0)
        elif phase < 0.5:
            vy = self.lateral_velocity_max
        elif phase < 0.8:
            vy = self.lateral_velocity_max * 0.95
        else:
            progress = (phase - 0.8) / 0.2
            vy = self.lateral_velocity_max * (0.95 - 0.15 * progress)
        
        # Vertical velocity from average leg length change
        avg_extension = (left_ext + right_ext) / 2.0
        if phase < 0.3:
            progress = phase / 0.3
            vz = self.extension_amplitude * (np.pi / 0.3) * np.sin(np.pi * progress) / (2.0 * self.freq)
        elif phase < 0.5:
            progress = (phase - 0.3) / 0.2
            vz = -self.extension_amplitude * (np.pi / 0.2) * np.sin(np.pi * progress) / (2.0 * self.freq)
        elif phase < 0.8:
            progress = (phase - 0.5) / 0.3
            vz = self.extension_amplitude * (np.pi / 0.3) * np.sin(np.pi * progress) / (2.0 * self.freq)
        else:
            progress = (phase - 0.8) / 0.2
            vz = -self.extension_amplitude * (np.pi / 0.2) * np.sin(np.pi * progress) / (2.0 * self.freq)
        
        vz *= 0.5  # Scale down for smoother motion

        # Set velocity commands (world frame)
        self.vel_world = np.array([0.0, vy, vz])
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
        Compute foot position in body frame maintaining ground contact.
        Extension increases the body-frame z-value (foot moves down relative to body)
        to maintain ground contact while effective leg length increases.
        """
        differential, left_ext, right_ext = self.compute_differential_extension(phase)
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name in self.left_legs
        
        # Apply extension as positive z offset in body frame
        # (foot moves "down" in body frame to stay on ground while leg extends)
        if is_left_leg:
            extension = left_ext
        else:
            extension = right_ext
        
        # Positive extension increases body-frame z (foot goes down relative to body center)
        # This keeps foot grounded while body rises on that side
        foot[2] += extension
        
        # Add small horizontal compensation to maintain stability during roll
        # As body rolls, adjust foot lateral position slightly to maintain contact geometry
        roll_compensation = 0.0
        if abs(differential) > 0.001:
            roll_angle_estimate = differential / (self.lateral_spacing + 0.1)
            if is_left_leg:
                roll_compensation = -0.02 * roll_angle_estimate
            else:
                roll_compensation = 0.02 * roll_angle_estimate
        
        foot[1] += roll_compensation
        
        return foot

    def get_target_positions(self, t):
        self.t = t
        phase = (t * self.freq) % 1.0
        
        dt = 1.0 / (self.freq * 100.0)
        self.update_base_motion(phase, dt)
        
        feet_pos_body = {}
        for leg in self.leg_names:
            feet_pos_body[leg] = self.compute_foot_position_body_frame(leg, phase)
        
        return feet_pos_body, self.root_pos, self.root_quat

    def get_current_velocities(self):
        return self.vel_world, self.omega_world