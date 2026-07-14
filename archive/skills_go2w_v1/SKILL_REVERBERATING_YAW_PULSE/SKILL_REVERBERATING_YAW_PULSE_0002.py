from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERBERATING_YAW_PULSE_MotionGenerator(BaseMotionGenerator):
    """
    Reverberating Yaw Pulse: In-place damped yaw oscillation with continuous four-leg ground contact.
    
    Motion consists of five damped yaw pulses:
      - Phase [0.0, 0.2]: 60° CCW
      - Phase [0.2, 0.4]: 40° CW
      - Phase [0.4, 0.6]: 25° CCW
      - Phase [0.6, 0.8]: 15° CW
      - Phase [0.8, 1.0]: 5° CCW
    
    Net rotation: ~45° counterclockwise.
    
    Legs modulate radial extension to generate/absorb angular momentum while maintaining ground contact.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Full cycle duration ~2 seconds
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time
        self.t = 0.0
        
        # Base state (WORLD frame)
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Yaw pulse parameters: (target_angle_deg, direction_sign)
        # Each tuple: (angular_displacement_deg, direction: +1=CCW, -1=CW)
        self.yaw_pulses = [
            (60.0, 1.0),   # Phase [0.0, 0.2]: 60° CCW
            (40.0, -1.0),  # Phase [0.2, 0.4]: 40° CW
            (25.0, 1.0),   # Phase [0.4, 0.6]: 25° CCW
            (15.0, -1.0),  # Phase [0.6, 0.8]: 15° CW
            (5.0, 1.0),    # Phase [0.8, 1.0]: 5° CCW
        ]
        
        # Radial extension amplitudes per phase (as fraction of nominal stance radius)
        # Positive = extend outward, negative = retract inward
        self.radial_extension_schedule = [
            0.12,   # Phase [0.0, 0.2]: extend for initial CCW pulse
            -0.08,  # Phase [0.2, 0.4]: retract for CW reversal
            0.06,   # Phase [0.4, 0.6]: moderate extend for CCW pulse
            -0.04,  # Phase [0.6, 0.8]: slight retract for CW correction
            0.0,    # Phase [0.8, 1.0]: return to neutral
        ]
        
        # Compute nominal stance radius for each leg (for radial modulation)
        self.nominal_radii = {}
        for leg in self.leg_names:
            base_pos = self.base_feet_pos_body[leg]
            self.nominal_radii[leg] = np.sqrt(base_pos[0]**2 + base_pos[1]**2)

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent yaw rate pulses.
        No translation, no roll/pitch.
        """
        # Determine which sub-phase we're in
        phase_idx = min(int(phase / 0.2), 4)
        local_phase = (phase - phase_idx * 0.2) / 0.2  # Normalize to [0,1] within sub-phase
        
        angle_deg, direction = self.yaw_pulses[phase_idx]
        angle_rad = np.deg2rad(angle_deg)
        
        # Smooth yaw rate profile: use sine wave for smooth acceleration/deceleration
        # Integral of sin over [0, pi] = 2, so scale appropriately
        # yaw_rate(t) = peak * sin(pi * local_phase)
        # Integral over local_phase [0,1] = peak * 2/pi
        # We want integral = angle_rad, so peak = angle_rad * pi / 2
        
        phase_duration = 0.2 / self.freq  # Duration of each sub-phase in seconds
        peak_yaw_rate = (angle_rad / phase_duration) * (np.pi / 2.0)
        
        yaw_rate = direction * peak_yaw_rate * np.sin(np.pi * local_phase)
        
        # Set velocities
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
        Feet remain grounded; position modulates radially to create/absorb angular momentum.
        """
        # Determine sub-phase
        phase_idx = min(int(phase / 0.2), 4)
        local_phase = (phase - phase_idx * 0.2) / 0.2  # Normalize to [0,1] within sub-phase
        
        # Get radial extension amplitude for this phase
        current_extension = self.radial_extension_schedule[phase_idx]
        
        # Smooth transition using cosine interpolation for radial extension
        # At start/end of sub-phase, blend with neighboring phases
        if phase_idx > 0:
            prev_extension = self.radial_extension_schedule[phase_idx - 1]
        else:
            prev_extension = 0.0
        
        if phase_idx < 4:
            next_extension = self.radial_extension_schedule[phase_idx + 1]
        else:
            next_extension = 0.0
        
        # Smooth interpolation within sub-phase using cosine taper
        blend = 0.5 * (1.0 - np.cos(np.pi * local_phase))
        if local_phase < 0.5:
            # Blend from previous to current
            extension_factor = prev_extension + (current_extension - prev_extension) * (2.0 * blend)
        else:
            # Blend from current to next
            extension_factor = current_extension + (next_extension - current_extension) * (2.0 * (blend - 0.5))
        
        # Get base foot position
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial direction (in xy plane)
        xy_radius = np.sqrt(base_pos[0]**2 + base_pos[1]**2)
        if xy_radius > 1e-6:
            radial_unit = np.array([base_pos[0] / xy_radius, base_pos[1] / xy_radius])
        else:
            radial_unit = np.array([1.0, 0.0])
        
        # Apply radial modulation
        radial_offset = extension_factor * self.nominal_radii[leg_name]
        foot_pos = base_pos.copy()
        foot_pos[0] += radial_offset * radial_unit[0]
        foot_pos[1] += radial_offset * radial_unit[1]
        
        # Z coordinate remains constant (grounded)
        # Optionally add slight z modulation for more realistic contact
        # Here we keep z constant for strict ground contact
        
        return foot_pos