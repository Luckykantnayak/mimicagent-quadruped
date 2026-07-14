from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_COMPASS_SWEEP_TURN_MotionGenerator(BaseMotionGenerator):
    """
    Compass Sweep Turn: In-place clockwise yaw rotation using alternating diagonal leg extension.
    
    Motion pattern:
    - Phase 0.0-0.25: FL+RR extend radially (compass arms), FR+RL tuck inward
    - Phase 0.25-0.5: All legs transition through neutral, base yaws
    - Phase 0.5-0.75: FR+RL extend radially (compass arms), FL+RR tuck inward
    - Phase 0.75-1.0: All legs return to neutral, base completes yaw increment
    
    All four feet maintain ground contact throughout the entire motion.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # 0.5 Hz for smooth, controlled rotation
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compute nominal radial distances for each leg
        self.nominal_radial_dist = {}
        for leg in self.leg_names:
            pos = self.base_feet_pos_body[leg]
            self.nominal_radial_dist[leg] = np.sqrt(pos[0]**2 + pos[1]**2)
        
        # Radial extension and tucking parameters
        self.extension_factor = 1.4  # Extended radius = 1.4x nominal
        self.tuck_factor = 0.6       # Tucked radius = 0.6x nominal
        
        # Yaw rate for clockwise rotation (positive = clockwise)
        # Target: ~45 degrees per cycle = π/4 radians per cycle
        # yaw_rate * (1/freq) = π/4 => yaw_rate = π/4 * freq
        self.yaw_rate = (np.pi / 4) * self.freq
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (will be set in update_base_motion)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Apply continuous clockwise yaw rotation with no translation.
        All phases maintain constant positive yaw rate.
        """
        # No translation - pure rotation in place
        self.vel_world = np.array([0.0, 0.0, 0.0])
        
        # Continuous clockwise yaw throughout all phases
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
        Compute foot position in body frame based on phase and leg-specific pattern.
        
        FL and RR (Group 1):
        - Phase 0.0-0.25: Extend radially outward
        - Phase 0.25-0.5: Transition from extended to neutral
        - Phase 0.5-0.75: Tuck inward toward body center
        - Phase 0.75-1.0: Transition from tucked to neutral
        
        FR and RL (Group 2):
        - Phase 0.0-0.25: Tuck inward toward body center
        - Phase 0.25-0.5: Transition from tucked to neutral
        - Phase 0.5-0.75: Extend radially outward
        - Phase 0.75-1.0: Transition from extended to neutral
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        nominal_radius = self.nominal_radial_dist[leg_name]
        
        # Compute angle to foot in body frame (for radial motion)
        angle_to_foot = np.arctan2(base_pos[1], base_pos[0])
        
        # Determine which diagonal group this leg belongs to
        is_group_1 = leg_name.startswith('FL') or leg_name.startswith('RR')
        
        # Compute target radial scale factor based on phase and group
        if is_group_1:
            # FL and RR: extend first half cycle, tuck second half
            radial_scale = self._compute_radial_scale_group1(phase)
        else:
            # FR and RL: tuck first half cycle, extend second half
            radial_scale = self._compute_radial_scale_group2(phase)
        
        # Apply radial scaling
        target_radius = nominal_radius * radial_scale
        
        # Compute new x, y based on radial distance
        foot_pos = base_pos.copy()
        foot_pos[0] = target_radius * np.cos(angle_to_foot)
        foot_pos[1] = target_radius * np.sin(angle_to_foot)
        # Z remains at ground level (base_pos[2])
        
        return foot_pos

    def _compute_radial_scale_group1(self, phase):
        """
        Compute radial scale for Group 1 (FL, RR).
        Phase 0.0-0.25: Extend (1.0 -> extension_factor)
        Phase 0.25-0.5: Return to neutral (extension_factor -> 1.0)
        Phase 0.5-0.75: Tuck (1.0 -> tuck_factor)
        Phase 0.75-1.0: Return to neutral (tuck_factor -> 1.0)
        """
        if phase < 0.25:
            # Extend phase: smooth interpolation from neutral to extended
            t = phase / 0.25
            return self._smooth_interpolate(1.0, self.extension_factor, t)
        elif phase < 0.5:
            # Transition back to neutral
            t = (phase - 0.25) / 0.25
            return self._smooth_interpolate(self.extension_factor, 1.0, t)
        elif phase < 0.75:
            # Tuck phase: smooth interpolation from neutral to tucked
            t = (phase - 0.5) / 0.25
            return self._smooth_interpolate(1.0, self.tuck_factor, t)
        else:
            # Transition back to neutral
            t = (phase - 0.75) / 0.25
            return self._smooth_interpolate(self.tuck_factor, 1.0, t)

    def _compute_radial_scale_group2(self, phase):
        """
        Compute radial scale for Group 2 (FR, RL).
        Phase 0.0-0.25: Tuck (1.0 -> tuck_factor)
        Phase 0.25-0.5: Return to neutral (tuck_factor -> 1.0)
        Phase 0.5-0.75: Extend (1.0 -> extension_factor)
        Phase 0.75-1.0: Return to neutral (extension_factor -> 1.0)
        """
        if phase < 0.25:
            # Tuck phase
            t = phase / 0.25
            return self._smooth_interpolate(1.0, self.tuck_factor, t)
        elif phase < 0.5:
            # Transition back to neutral
            t = (phase - 0.25) / 0.25
            return self._smooth_interpolate(self.tuck_factor, 1.0, t)
        elif phase < 0.75:
            # Extend phase
            t = (phase - 0.5) / 0.25
            return self._smooth_interpolate(1.0, self.extension_factor, t)
        else:
            # Transition back to neutral
            t = (phase - 0.75) / 0.25
            return self._smooth_interpolate(self.extension_factor, 1.0, t)

    def _smooth_interpolate(self, start, end, t):
        """
        Smooth interpolation using cubic easing (ease-in-out).
        t ∈ [0, 1]
        """
        # Cubic ease-in-out for smooth acceleration/deceleration
        if t < 0.5:
            smooth_t = 4 * t * t * t
        else:
            smooth_t = 1 - pow(-2 * t + 2, 3) / 2
        
        return start + (end - start) * smooth_t