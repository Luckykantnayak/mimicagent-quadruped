from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_KAYAK_PADDLE_TURN_MotionGenerator(BaseMotionGenerator):
    """
    Kayak paddle turn: in-place yaw rotation via asymmetric leg sweeps.
    
    Left and right leg pairs alternate between power strokes (backward sweeps)
    and recovery strokes (forward returns), creating rotational torque while
    maintaining continuous ground contact throughout the entire cycle.
    
    All four feet remain in contact with the ground at all times, sliding
    horizontally in arc-like paddle motions.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for deliberate paddling motion
        
        # Base foot positions in BODY frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Paddle motion parameters
        self.arc_length_x = 0.15  # Forward/backward sweep distance
        self.arc_width_y = 0.08   # Lateral sweep distance (in/out)
        self.ground_clearance = 0.005  # Minimal z offset for sliding contact
        
        # Yaw rotation parameters
        self.yaw_rate_magnitude = 0.8  # rad/s during active paddling
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base to produce in-place yaw rotation.
        Linear velocities remain zero, yaw rate varies with phase.
        """
        # No linear translation - stay in place
        vx, vy, vz = 0.0, 0.0, 0.0
        
        # Yaw rate modulation across phases
        if phase < 0.25:
            # Left power, right recovery - full yaw rate
            yaw_rate = self.yaw_rate_magnitude
        elif phase < 0.5:
            # Transition 1 - smooth modulation
            t_local = (phase - 0.25) / 0.25
            yaw_rate = self.yaw_rate_magnitude * (1.0 - 0.3 * np.sin(np.pi * t_local))
        elif phase < 0.75:
            # Right power, left recovery - full yaw rate
            yaw_rate = self.yaw_rate_magnitude
        else:
            # Transition 2 - smooth modulation back to start
            t_local = (phase - 0.75) / 0.25
            yaw_rate = self.yaw_rate_magnitude * (1.0 - 0.3 * np.sin(np.pi * t_local))
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot trajectory in BODY frame.
        
        Left legs (FL, RL): power stroke [0, 0.5), recovery [0.5, 1.0)
        Right legs (FR, RR): recovery [0, 0.5), power stroke [0.5, 1.0)
        
        Power stroke: sweep backward and outward
        Recovery stroke: return forward and inward
        All motion stays close to ground (continuous contact)
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is on left or right side
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Lateral sign: left legs move outward (negative y), right legs outward (positive y)
        lateral_sign = -1.0 if is_left else 1.0
        
        if is_left:
            # Left legs: power [0, 0.5), recovery [0.5, 1.0)
            if phase < 0.5:
                # Power and transition 1
                t_local = phase / 0.5
                dx, dy = self._compute_power_stroke_offset(t_local, lateral_sign)
            else:
                # Recovery and transition 2
                t_local = (phase - 0.5) / 0.5
                dx, dy = self._compute_recovery_stroke_offset(t_local, lateral_sign)
        else:
            # Right legs: recovery [0, 0.5), power [0.5, 1.0)
            if phase < 0.5:
                # Recovery and transition 1
                t_local = phase / 0.5
                dx, dy = self._compute_recovery_stroke_offset(t_local, lateral_sign)
            else:
                # Power and transition 2
                t_local = (phase - 0.5) / 0.5
                dx, dy = self._compute_power_stroke_offset(t_local, lateral_sign)
        
        # Apply offsets to base position
        foot = base_pos.copy()
        foot[0] += dx
        foot[1] += dy
        foot[2] = -np.abs(base_pos[2]) + self.ground_clearance  # Stay near ground
        
        return foot

    def _compute_power_stroke_offset(self, t, lateral_sign):
        """
        Power stroke: sweep from forward-inward to backward-outward.
        
        t ∈ [0, 1] maps the full power stroke trajectory
        lateral_sign: -1 for left legs, +1 for right legs
        """
        # Use smooth interpolation for arc-like motion
        # X: forward (+) to backward (-)
        dx = self.arc_length_x * (0.5 - t)
        
        # Y: inward (0) to outward (lateral_sign * arc_width_y)
        # Smooth curve for natural paddling motion
        dy = lateral_sign * self.arc_width_y * np.sin(np.pi * t / 2.0)
        
        return dx, dy

    def _compute_recovery_stroke_offset(self, t, lateral_sign):
        """
        Recovery stroke: return from backward-outward to forward-inward.
        
        t ∈ [0, 1] maps the full recovery stroke trajectory
        lateral_sign: -1 for left legs, +1 for right legs
        """
        # X: backward (-) to forward (+)
        dx = self.arc_length_x * (-0.5 + t)
        
        # Y: outward (lateral_sign * arc_width_y) to inward (0)
        # Smooth return trajectory
        dy = lateral_sign * self.arc_width_y * np.cos(np.pi * t / 2.0)
        
        return dx, dy