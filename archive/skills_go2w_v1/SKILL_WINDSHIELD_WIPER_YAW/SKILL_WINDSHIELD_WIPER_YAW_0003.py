from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_WINDSHIELD_WIPER_YAW_MotionGenerator(BaseMotionGenerator):
    """
    Windshield wiper yaw oscillation: asymmetric in-place rotation.
    
    Phase structure:
      [0.0, 0.6]: Slow clockwise yaw sweep (~50 degrees accumulated)
      [0.6, 0.7]: Fast counterclockwise yaw return (reverse ~50 degrees)
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
        
        # Determine nominal base height from initial foot positions
        # Assumes feet should be on ground (world z=0) at initialization
        avg_foot_z_body = np.mean([v[2] for v in initial_foot_positions_body.values()])
        self.nominal_base_height = -avg_foot_z_body
        if self.nominal_base_height < 0.4:
            self.nominal_base_height = 0.5  # Safe fallback for typical quadruped
        
        # Yaw motion parameters (reduced from 70 to 50 degrees to respect joint limits)
        self.target_yaw_displacement = np.deg2rad(50.0)
        
        # Phase boundaries
        self.sweep_end = 0.6
        self.return_end = 0.7
        self.pause_end = 1.0
        
        # Compute yaw rates to achieve target displacement
        sweep_duration_per_cycle = self.sweep_end
        self.yaw_rate_sweep = self.target_yaw_displacement / (sweep_duration_per_cycle / self.freq)
        
        return_duration_per_cycle = (self.return_end - self.sweep_end)
        self.yaw_rate_return = -self.target_yaw_displacement / (return_duration_per_cycle / self.freq)
        
        # Radial retraction parameters to keep feet within workspace during rotation
        self.max_radial_retraction = 0.05  # Maximum inward adjustment (meters)
        
        # Base height modulation to relieve lateral joint strain at peak yaw
        self.max_height_increase = 0.03  # Maximum base lift (meters)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.nominal_base_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track accumulated yaw for leg body-frame adjustment
        self.accumulated_yaw = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base yaw according to phase-dependent yaw rate profile.
        No translation: robot remains in place.
        Base height modulated to reduce joint strain during peak rotation.
        """
        # Determine yaw rate based on phase with smooth transitions
        if phase < self.sweep_end:
            # Slow clockwise sweep
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
        
        # Compute base height adjustment based on yaw magnitude to relieve joint stress
        height_adjustment = self._compute_height_adjustment(phase)
        
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
        
        # Set base height to nominal height plus modulation (not accumulated)
        # This ensures height oscillates around nominal rather than drifting
        self.root_pos[2] = self.nominal_base_height + height_adjustment
        
        # Track accumulated yaw for body-frame leg adjustments
        self.accumulated_yaw += yaw_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with radial retraction during rotation.
        
        Legs maintain ground contact and adjust their body-frame positions
        to accommodate the yawing base. Feet are pulled slightly inward during
        peak yaw to keep joints within operational limits.
        """
        # Get base foot position
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        # Compute phase-dependent yaw offset
        if phase < self.sweep_end:
            # During slow sweep: accumulated yaw grows from 0 to target_yaw_displacement
            phase_progress = phase / self.sweep_end
            yaw_offset = self.target_yaw_displacement * self._smooth_step(phase_progress)
            
        elif phase < self.return_end:
            # During fast return: yaw decreases from target back to ~0
            phase_progress = (phase - self.sweep_end) / (self.return_end - self.sweep_end)
            yaw_offset = self.target_yaw_displacement * (1.0 - self._smooth_step(phase_progress))
            
        else:
            # During pause: yaw offset is zero
            yaw_offset = 0.0
        
        # Compute radial retraction factor based on yaw magnitude
        # Maximum retraction at peak yaw (phase ~0.6)
        yaw_normalized = abs(yaw_offset) / self.target_yaw_displacement if self.target_yaw_displacement > 0 else 0.0
        retraction_factor = self._smooth_step(yaw_normalized)
        
        # Apply radial retraction: pull foot toward body center
        foot_horizontal = np.array([foot_base[0], foot_base[1]])
        foot_radius = np.linalg.norm(foot_horizontal)
        
        if foot_radius > 1e-6:
            # Reduce radius by retraction amount
            retracted_radius = foot_radius - (self.max_radial_retraction * retraction_factor)
            retracted_radius = max(retracted_radius, foot_radius * 0.7)  # Limit max retraction
            radial_scale = retracted_radius / foot_radius
            foot_horizontal = foot_horizontal * radial_scale
        
        # Rotate the retracted foot position by negative yaw offset
        foot_adjusted = self._rotate_point_2d(
            foot_horizontal[0], 
            foot_horizontal[1], 
            -yaw_offset
        )
        
        # Maintain z coordinate (vertical position unchanged in body frame)
        foot = np.array([foot_adjusted[0], foot_adjusted[1], foot_base[2]])
        
        return foot

    def _compute_height_adjustment(self, phase):
        """
        Compute base height adjustment to relieve joint strain during peak rotation.
        Height increases smoothly during maximum yaw displacement phases.
        Returns absolute height adjustment to be added to nominal height.
        """
        # Compute yaw magnitude based on phase
        if phase < self.sweep_end:
            phase_progress = phase / self.sweep_end
            yaw_normalized = self._smooth_step(phase_progress)
            
        elif phase < self.return_end:
            phase_progress = (phase - self.sweep_end) / (self.return_end - self.sweep_end)
            yaw_normalized = 1.0 - self._smooth_step(phase_progress)
            
        else:
            yaw_normalized = 0.0
        
        # Apply smooth height increase proportional to yaw magnitude
        # Peak height at phase ~0.6 when yaw is maximum
        # Use symmetric envelope to prevent bias accumulation
        height_adjustment = self.max_height_increase * yaw_normalized * (1.0 - yaw_normalized * 0.3)
        
        return height_adjustment

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