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

    Leg motion is purely vertical (z-axis modulation in body frame).
    Base motion prescribes lateral velocity and roll oscillations.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz) - slower for stability

        # Vertical extension parameters
        self.extension_amplitude = 0.06  # Max vertical extension (m)
        
        # Lateral velocity parameters
        self.lateral_velocity_max = 0.15  # Max rightward velocity (m/s)
        
        # Roll oscillation parameters
        self.roll_rate_max = 0.4  # Max roll rate (rad/s)

        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

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

    def update_base_motion(self, phase, dt):
        """
        Update base motion with lateral velocity and roll oscillations.
        
        Phase-based behavior:
          [0.0, 0.3]: Rightward tilt develops (negative roll rate), lateral velocity increases
          [0.3, 0.5]: Base levels (positive roll rate), lateral velocity maintained
          [0.5, 0.8]: Leftward tilt develops (positive roll rate), lateral velocity sustained
          [0.8, 1.0]: Base returns to neutral (negative roll rate), lateral velocity maintained
        """
        
        # Compute lateral velocity (vy) - positive throughout, varying amplitude
        if phase < 0.3:
            # Increasing lateral velocity as tilt develops
            progress = phase / 0.3
            vy = self.lateral_velocity_max * (0.5 + 0.5 * progress)
        elif phase < 0.5:
            # Maintain velocity during equalization
            vy = self.lateral_velocity_max
        elif phase < 0.8:
            # Sustained velocity during opposite tilt
            vy = self.lateral_velocity_max * 0.9
        else:
            # Slight decay in final equalization
            progress = (phase - 0.8) / 0.2
            vy = self.lateral_velocity_max * (0.9 - 0.1 * progress)
        
        # Compute roll rate
        if phase < 0.3:
            # Negative roll rate (tilting right side down)
            progress = phase / 0.3
            roll_rate = -self.roll_rate_max * np.sin(np.pi * progress)
        elif phase < 0.5:
            # Positive roll rate (leveling from right tilt)
            progress = (phase - 0.3) / 0.2
            roll_rate = self.roll_rate_max * np.sin(np.pi * progress)
        elif phase < 0.8:
            # Positive roll rate (tilting left side down)
            progress = (phase - 0.5) / 0.3
            roll_rate = self.roll_rate_max * np.sin(np.pi * progress)
        else:
            # Negative roll rate (leveling from left tilt)
            progress = (phase - 0.8) / 0.2
            roll_rate = -self.roll_rate_max * np.sin(np.pi * progress)
        
        # Small vertical velocity oscillation due to net leg extension changes
        if phase < 0.3:
            vz = 0.02 * np.sin(np.pi * phase / 0.3)
        elif phase < 0.5:
            vz = -0.01 * np.sin(np.pi * (phase - 0.3) / 0.2)
        elif phase < 0.8:
            vz = -0.02 * np.sin(np.pi * (phase - 0.5) / 0.3)
        else:
            vz = 0.01 * np.sin(np.pi * (phase - 0.8) / 0.2)

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
        Compute foot position in body frame with vertical (z-axis) modulation.
        
        Left legs (FL, RL):
          [0.0, 0.3]: Extend (z decreases)
          [0.3, 0.5]: Return to neutral
          [0.5, 0.8]: Compress (z increases)
          [0.8, 1.0]: Return to neutral
        
        Right legs (FR, RR):
          [0.0, 0.3]: Compress (z increases)
          [0.3, 0.5]: Return to neutral
          [0.5, 0.8]: Extend (z decreases)
          [0.8, 1.0]: Return to neutral
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name in self.left_legs
        
        # Compute vertical offset based on phase
        if phase < 0.3:
            # Phase 1: Left extend, right compress
            progress = phase / 0.3
            extension = self.extension_amplitude * np.sin(np.pi * progress)
            if is_left_leg:
                foot[2] -= extension  # Extend (decrease z in body frame)
            else:
                foot[2] += extension  # Compress (increase z in body frame)
        
        elif phase < 0.5:
            # Phase 2: Equalization
            progress = (phase - 0.3) / 0.2
            # Smooth transition from peak extension back to neutral
            extension = self.extension_amplitude * np.sin(np.pi * (1.0 - progress))
            if is_left_leg:
                foot[2] -= extension
            else:
                foot[2] += extension
        
        elif phase < 0.8:
            # Phase 3: Right extend, left compress
            progress = (phase - 0.5) / 0.3
            extension = self.extension_amplitude * np.sin(np.pi * progress)
            if is_left_leg:
                foot[2] += extension  # Compress
            else:
                foot[2] -= extension  # Extend
        
        else:
            # Phase 4: Final equalization
            progress = (phase - 0.8) / 0.2
            extension = self.extension_amplitude * np.sin(np.pi * (1.0 - progress))
            if is_left_leg:
                foot[2] += extension
            else:
                foot[2] -= extension
        
        return foot