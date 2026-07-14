from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_FOLDING_FAN_ROTATION_MotionGenerator(BaseMotionGenerator):
    """
    Folding Fan Rotation: In-place yaw rotation with alternating leg fold/unfold pattern.
    
    - Front legs (FL, FR) fold inward during phase [0.0, 0.25], extend during [0.25, 0.5],
      hold extended [0.5, 0.75], then fold again [0.75, 1.0].
    - Rear legs (RL, RR) extend outward during phase [0.0, 0.25], fold during [0.25, 0.5],
      hold folded [0.5, 0.75], then extend again [0.75, 1.0].
    - Base maintains constant positive yaw rate throughout entire cycle.
    - All four feet remain grounded (stance) at all times.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0  # 1 Hz cycle frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Fold/unfold amplitude (lateral displacement in body frame y-axis)
        self.fold_amplitude = 0.15  # meters, inward/outward movement
        
        # Constant yaw rate (rad/s) - moderate for stability
        self.yaw_rate = 0.4  # radians per second, clockwise rotation
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (set in update_base_motion)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Constant clockwise yaw rotation with zero linear velocity.
        """
        # No linear velocity (in-place rotation)
        self.vel_world = np.array([0.0, 0.0, 0.0])
        
        # Constant positive yaw rate (clockwise rotation)
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate])
        
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
        Compute foot position in body frame based on phase.
        
        Front legs (FL, FR):
          [0.00, 0.25]: fold inward (y → y - fold_amplitude)
          [0.25, 0.50]: unfold outward (y → y + fold_amplitude)
          [0.50, 0.75]: hold extended
          [0.75, 1.00]: fold inward again
        
        Rear legs (RL, RR):
          [0.00, 0.25]: unfold outward (y → y + fold_amplitude)
          [0.25, 0.50]: fold inward (y → y - fold_amplitude)
          [0.50, 0.75]: hold folded
          [0.75, 1.00]: unfold outward again
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine lateral sign based on leg side (FL, RL = left, positive y; FR, RR = right, negative y)
        if leg_name.endswith('L'):
            lateral_sign = 1.0  # Left legs have positive y in body frame
        else:
            lateral_sign = -1.0  # Right legs have negative y in body frame
        
        # Front legs (FL, FR)
        if leg_name.startswith('F'):
            if phase < 0.25:
                # Fold inward: smooth transition from 0 to -fold_amplitude
                progress = phase / 0.25
                offset = -self.fold_amplitude * self._smooth_step(progress)
            elif phase < 0.5:
                # Unfold outward: smooth transition from -fold_amplitude to +fold_amplitude
                progress = (phase - 0.25) / 0.25
                offset = -self.fold_amplitude + 2.0 * self.fold_amplitude * self._smooth_step(progress)
            elif phase < 0.75:
                # Hold extended at +fold_amplitude
                offset = self.fold_amplitude
            else:
                # Fold inward again: smooth transition from +fold_amplitude to 0
                progress = (phase - 0.75) / 0.25
                offset = self.fold_amplitude * (1.0 - self._smooth_step(progress))
        
        # Rear legs (RL, RR)
        else:
            if phase < 0.25:
                # Unfold outward: smooth transition from 0 to +fold_amplitude
                progress = phase / 0.25
                offset = self.fold_amplitude * self._smooth_step(progress)
            elif phase < 0.5:
                # Fold inward: smooth transition from +fold_amplitude to -fold_amplitude
                progress = (phase - 0.25) / 0.25
                offset = self.fold_amplitude - 2.0 * self.fold_amplitude * self._smooth_step(progress)
            elif phase < 0.75:
                # Hold folded at -fold_amplitude
                offset = -self.fold_amplitude
            else:
                # Unfold outward again: smooth transition from -fold_amplitude to 0
                progress = (phase - 0.75) / 0.25
                offset = -self.fold_amplitude + self.fold_amplitude * self._smooth_step(progress)
        
        # Apply lateral offset with correct sign for left/right legs
        foot[1] += lateral_sign * offset
        
        # Z remains at ground level (no vertical motion)
        return foot

    def _smooth_step(self, x):
        """
        Smooth interpolation function using cubic easing (smoothstep).
        Maps [0, 1] → [0, 1] with zero derivatives at endpoints.
        """
        return x * x * (3.0 - 2.0 * x)