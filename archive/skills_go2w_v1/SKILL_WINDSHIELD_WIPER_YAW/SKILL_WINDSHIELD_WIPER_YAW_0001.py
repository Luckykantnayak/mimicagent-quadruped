from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_WINDSHIELD_WIPER_YAW_MotionGenerator(BaseMotionGenerator):
    """
    Windshield wiper yaw oscillation: asymmetric in-place rotation.
    
    Phase structure:
      [0.0, 0.6]: Slow clockwise yaw sweep (~70 degrees accumulated)
      [0.6, 0.7]: Fast counterclockwise yaw return (reverse ~70 degrees)
      [0.7, 1.0]: Pause phase (zero yaw rate, stabilization)
    
    All four legs remain in continuous ground contact throughout.
    Legs adjust their body-frame positions to accommodate base yaw rotation
    while maintaining stationary world-frame foot positions (kinematic in-place rotation).
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Cycle frequency (Hz)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Yaw motion parameters
        # Target: ~70 degrees = ~1.22 radians total yaw displacement
        # Slow sweep over 60% of cycle, fast return over 10% of cycle
        self.target_yaw_displacement = np.deg2rad(70.0)
        
        # Phase boundaries
        self.sweep_end = 0.6
        self.return_end = 0.7
        self.pause_end = 1.0
        
        # Compute yaw rates to achieve target displacement
        # During sweep [0, 0.6]: duration = 0.6 / freq
        # yaw_rate_sweep * duration = target_yaw_displacement
        sweep_duration_per_cycle = self.sweep_end
        self.yaw_rate_sweep = self.target_yaw_displacement / (sweep_duration_per_cycle / self.freq)
        
        # During return [0.6, 0.7]: duration = 0.1 / freq
        # yaw_rate_return * duration = -target_yaw_displacement
        return_duration_per_cycle = (self.return_end - self.sweep_end)
        self.yaw_rate_return = -self.target_yaw_displacement / (return_duration_per_cycle / self.freq)
        
        # Leg adjustment parameters
        # As base yaws, legs appear to rotate in body frame to maintain world position
        # We model this as a circular arc adjustment in body frame x-y plane
        self.leg_adjustment_radius = 0.05  # Maximum radial adjustment for legs during yaw
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track accumulated yaw for leg body-frame adjustment
        self.accumulated_yaw = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base yaw according to phase-dependent yaw rate profile.
        No translation: robot remains in place.
        """
        # Determine yaw rate based on phase with smooth transitions
        if phase < self.sweep_end:
            # Slow clockwise sweep
            # Apply smooth ramp at start and end to reduce jerk
            phase_progress = phase / self.sweep_end
            smoothing = self._smooth_step(phase_progress)
            yaw_rate = self.yaw_rate_sweep * smoothing
            
        elif phase < self.return_end:
            # Fast counterclockwise return
            phase_progress = (phase - self.sweep_end) / (self.return_end - self.sweep_end)
            smoothing = self._smooth_step(phase_progress)
            yaw_rate = self.yaw_rate_return * smoothing
            
        else:
            # Pause phase: zero yaw rate
            yaw_rate = 0.0
        
        # Set velocity commands (zero translation, yaw rotation only)
        self.vel_world = np.array([0.0, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Track accumulated yaw for body-frame leg adjustments
        self.accumulated_yaw += yaw_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame.
        
        Legs maintain ground contact and adjust their body-frame positions
        to accommodate the yawing base. As the base rotates, each foot's
        body-frame coordinates rotate to keep the foot stationary in world frame.
        
        We model this by rotating the base foot position by the negative of
        the phase-dependent yaw angle (opposite rotation to maintain world position).
        """
        # Get base foot position
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        # Compute phase-dependent yaw offset
        # This represents how much the body has rotated relative to cycle start
        if phase < self.sweep_end:
            # During slow sweep: accumulated yaw grows from 0 to target_yaw_displacement
            phase_progress = phase / self.sweep_end
            yaw_offset = self.target_yaw_displacement * self._smooth_step(phase_progress)
            
        elif phase < self.return_end:
            # During fast return: yaw decreases from target back to ~0
            phase_progress = (phase - self.sweep_end) / (self.return_end - self.sweep_end)
            yaw_offset = self.target_yaw_displacement * (1.0 - self._smooth_step(phase_progress))
            
        else:
            # During pause: yaw offset is zero (body returned to near-initial orientation)
            yaw_offset = 0.0
        
        # To keep foot stationary in world frame while body yaws,
        # rotate foot position in body frame by negative yaw offset
        foot_adjusted = self._rotate_point_2d(
            foot_base[0], 
            foot_base[1], 
            -yaw_offset
        )
        
        # Maintain z coordinate (vertical position unchanged)
        foot = np.array([foot_adjusted[0], foot_adjusted[1], foot_base[2]])
        
        return foot

    def _rotate_point_2d(self, x, y, angle):
        """
        Rotate a 2D point (x, y) by angle (radians) around origin.
        Returns rotated (x', y') as numpy array.
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x_new = cos_a * x - sin_a * y
        y_new = sin_a * x + cos_a * y
        return np.array([x_new, y_new])

    def _smooth_step(self, t):
        """
        Smooth step function for reducing jerk at phase transitions.
        Uses smoothstep: 3t^2 - 2t^3
        Input t in [0, 1], output smoothly interpolates from 0 to 1.
        """
        t = np.clip(t, 0.0, 1.0)
        return 3.0 * t**2 - 2.0 * t**3