from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERBERATING_YAW_PULSE_MotionGenerator(BaseMotionGenerator):
    """
    Reverberating Yaw Pulse: In-place damped oscillatory yaw rotation.
    
    The robot executes a series of five yaw pulses with alternating direction
    and exponentially decaying magnitude. Each pulse integrates to a specific
    angular displacement: 60°, -40°, 25°, -15°, 5° (CCW positive), resulting
    in a net 45° counterclockwise rotation.
    
    All four feet remain grounded throughout. Legs modulate radial extension
    synchronously with yaw pulse amplitude to manage reaction torques.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0  # One complete cycle per second

        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {
            k: v.copy() for k, v in initial_foot_positions_body.items()
        }

        # Base state (WORLD frame)
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Phase boundaries and target angular displacements
        self.phase_boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Target yaw displacements per phase (degrees)
        self.yaw_deltas_deg = [60.0, -40.0, 25.0, -15.0, 5.0]
        
        # Convert to radians for computation
        self.yaw_deltas_rad = [np.deg2rad(d) for d in self.yaw_deltas_deg]
        
        # Radial extension decay factors per phase
        # Maximum extension in phase 0, then exponentially decay
        self.extension_factors = [1.0, 0.67, 0.42, 0.25, 0.08]
        
        # Maximum radial extension distance (meters)
        self.max_radial_extension = 0.08

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent yaw rate commands.
        Linear velocities remain zero (in-place rotation).
        """
        # Determine which phase segment we're in
        phase_idx = self._get_phase_index(phase)
        
        # Compute local phase within current segment [0, 1]
        local_phase = self._get_local_phase(phase, phase_idx)
        
        # Compute smooth yaw rate for current phase segment
        yaw_rate = self._compute_yaw_rate(phase_idx, local_phase)
        
        # Set velocity commands (zero linear, yaw rate only)
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

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in BODY frame with radial extension modulation.
        
        Legs extend radially during high-amplitude yaw pulses and retract
        toward nominal stance as oscillation dampens. Extension amplitude
        decays with yaw pulse magnitude.
        """
        # Get nominal foot position
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        # Determine current phase segment
        phase_idx = self._get_phase_index(phase)
        local_phase = self._get_local_phase(phase, phase_idx)
        
        # Compute radial extension for this phase
        extension_factor = self.extension_factors[phase_idx]
        
        # Smooth extension profile within phase (ramp up then down)
        # Use sine envelope to avoid discontinuities at phase boundaries
        phase_envelope = np.sin(np.pi * local_phase)
        
        # Compute radial direction in xy plane
        radial_xy = np.linalg.norm(foot_base[:2])
        if radial_xy > 1e-6:
            radial_direction = foot_base[:2] / radial_xy
        else:
            radial_direction = np.array([1.0, 0.0])
        
        # Apply radial extension (scaled by decay factor and envelope)
        radial_offset = (
            self.max_radial_extension * 
            extension_factor * 
            phase_envelope
        )
        
        # Adjust foot position
        foot = foot_base.copy()
        foot[:2] += radial_direction * radial_offset
        
        # Front legs adjust slightly forward, rear legs slightly rearward
        # to maintain symmetric torque distribution
        if leg_name.startswith('F'):
            foot[0] += 0.02 * extension_factor * phase_envelope
        else:  # Rear legs
            foot[0] -= 0.02 * extension_factor * phase_envelope
        
        return foot

    def _get_phase_index(self, phase):
        """
        Determine which of the 5 phase segments the current phase belongs to.
        Returns index 0-4.
        """
        for i in range(len(self.phase_boundaries) - 1):
            if phase >= self.phase_boundaries[i] and phase < self.phase_boundaries[i + 1]:
                return i
        # Handle phase == 1.0 edge case
        return len(self.yaw_deltas_rad) - 1

    def _get_local_phase(self, phase, phase_idx):
        """
        Compute normalized local phase [0, 1] within current segment.
        """
        phase_start = self.phase_boundaries[phase_idx]
        phase_end = self.phase_boundaries[phase_idx + 1]
        phase_duration = phase_end - phase_start
        
        if phase_duration > 1e-9:
            return (phase - phase_start) / phase_duration
        else:
            return 0.0

    def _compute_yaw_rate(self, phase_idx, local_phase):
        """
        Compute smooth yaw rate for current phase segment.
        
        Yaw rate is shaped with a smooth profile to integrate to target
        angular displacement over the phase duration, while avoiding
        discontinuities at phase boundaries.
        """
        # Phase duration (each segment is 0.2 of full phase)
        phase_duration = 0.2
        
        # Target angular displacement for this phase segment
        target_angle = self.yaw_deltas_rad[phase_idx]
        
        # Use smooth acceleration profile (sine-based)
        # This ensures smooth ramp-up and ramp-down at phase boundaries
        # Integral of sin(pi*t) from 0 to 1 is 2/pi
        # To integrate to target_angle, scale by target_angle * pi/2
        profile = np.sin(np.pi * local_phase)
        
        # Yaw rate magnitude (rad/s)
        # Average rate needed: target_angle / phase_duration
        # With sine profile, peak rate is (pi/2) * average_rate
        avg_rate = target_angle / phase_duration
        yaw_rate = (np.pi / 2.0) * avg_rate * profile
        
        return yaw_rate