from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERBERATING_YAW_PULSE_MotionGenerator(BaseMotionGenerator):
    """
    Reverberating yaw pulse skill with damped oscillatory angular motion.
    
    The robot remains on all four feet throughout, executing alternating
    counterclockwise and clockwise yaw rate commands with decreasing magnitude.
    Legs modulate radial extension to assist angular momentum generation and damping.
    
    Net result: ~45-degree counterclockwise rotation after one full cycle.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compute nominal radial distances for each leg
        self.nominal_radii = {}
        for leg in self.leg_names:
            pos = self.base_feet_pos_body[leg]
            self.nominal_radii[leg] = np.sqrt(pos[0]**2 + pos[1]**2)
        
        # Radial modulation amplitude (fraction of nominal radius)
        self.radial_modulation = 0.10
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Define phase boundaries
        self.phase_boundaries = [
            (0.0, 0.2),   # pulse_1_ccw_primary
            (0.2, 0.4),   # pulse_2_cw_rebound
            (0.4, 0.6),   # pulse_3_ccw_secondary
            (0.6, 0.8),   # pulse_4_cw_tertiary
            (0.8, 1.0),   # pulse_5_ccw_settling
        ]
        
        # Target angular displacements for each pulse (degrees)
        self.angular_displacements = [60.0, -40.0, 25.0, -15.0, 5.0]
        
        # Convert to radians
        self.angular_displacements_rad = [np.deg2rad(d) for d in self.angular_displacements]
        
        # Compute average yaw rates for each phase
        # yaw_rate = angular_displacement / phase_duration
        self.yaw_rates = []
        for i, (start, end) in enumerate(self.phase_boundaries):
            phase_duration = end - start
            time_duration = phase_duration / self.freq
            yaw_rate = self.angular_displacements_rad[i] / time_duration
            self.yaw_rates.append(yaw_rate)
        
        # Radial extension profiles (multipliers relative to nominal)
        # Positive for counterclockwise (extend), negative for clockwise (retract)
        self.radial_profiles = [
            1.0,   # pulse_1: extend
            -0.67, # pulse_2: retract (scaled by relative magnitude)
            0.42,  # pulse_3: extend (scaled)
            -0.25, # pulse_4: retract (scaled)
            0.08,  # pulse_5: slight extend (scaled)
        ]

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent yaw rate with smooth transitions.
        No linear translation (in-place rotation).
        """
        # Determine which phase we're in and compute smooth yaw rate
        yaw_rate = self._compute_yaw_rate(phase)
        
        # Set velocities (no linear motion, only yaw rotation)
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

    def _compute_yaw_rate(self, phase):
        """
        Compute yaw rate for current phase with smooth transitions.
        Uses sinusoidal ramping at phase boundaries to avoid discontinuities.
        """
        # Find current phase index
        phase_idx = 0
        for i, (start, end) in enumerate(self.phase_boundaries):
            if start <= phase < end:
                phase_idx = i
                break
        
        # Local phase within current sub-phase [0, 1]
        start, end = self.phase_boundaries[phase_idx]
        local_phase = (phase - start) / (end - start)
        
        # Base yaw rate for this phase
        base_yaw_rate = self.yaw_rates[phase_idx]
        
        # Apply smooth envelope (sinusoidal ramp up/down at boundaries)
        # Ramp over 20% of phase duration at each end
        ramp_fraction = 0.2
        
        if local_phase < ramp_fraction:
            # Ramp up
            envelope = 0.5 * (1.0 - np.cos(np.pi * local_phase / ramp_fraction))
        elif local_phase > (1.0 - ramp_fraction):
            # Ramp down
            envelope = 0.5 * (1.0 + np.cos(np.pi * (local_phase - (1.0 - ramp_fraction)) / ramp_fraction))
        else:
            # Full magnitude in middle
            envelope = 1.0
        
        return base_yaw_rate * envelope

    def _compute_radial_modulation(self, phase):
        """
        Compute radial extension multiplier for current phase.
        Returns value in range [-1, 1] scaled by modulation amplitude.
        """
        # Find current phase index
        phase_idx = 0
        for i, (start, end) in enumerate(self.phase_boundaries):
            if start <= phase < end:
                phase_idx = i
                break
        
        # Local phase within current sub-phase [0, 1]
        start, end = self.phase_boundaries[phase_idx]
        local_phase = (phase - start) / (end - start)
        
        # Base radial profile for this phase
        base_radial = self.radial_profiles[phase_idx]
        
        # Apply smooth envelope (sinusoidal)
        envelope = np.sin(np.pi * local_phase)
        
        return base_radial * envelope * self.radial_modulation

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in BODY frame with radial modulation.
        All feet remain on ground (constant z), but modulate x-y position radially.
        """
        # Get base foot position
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial modulation factor
        radial_mod = self._compute_radial_modulation(phase)
        
        # Compute direction vector (normalized x-y projection)
        xy_norm = np.sqrt(base_pos[0]**2 + base_pos[1]**2)
        if xy_norm > 1e-6:
            direction = np.array([base_pos[0] / xy_norm, base_pos[1] / xy_norm])
        else:
            direction = np.array([1.0, 0.0])
        
        # Apply radial modulation to x-y position
        nominal_radius = self.nominal_radii[leg_name]
        modulated_radius = nominal_radius * (1.0 + radial_mod)
        
        foot_pos = base_pos.copy()
        foot_pos[0] = direction[0] * modulated_radius
        foot_pos[1] = direction[1] * modulated_radius
        # foot_pos[2] remains unchanged (ground contact)
        
        return foot_pos