from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_LATERAL_COIL_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Lateral Coil Spin: In-place 360-degree yaw rotation with progressive
    radial leg retraction and base rise.
    
    All four feet maintain continuous ground contact throughout the cycle.
    Legs retract symmetrically toward body centerline as base spins clockwise
    and rises vertically.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Lower frequency for smooth, controlled coiling motion
        
        # Store initial foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compute initial radial distances for each leg (horizontal distance from body center)
        self.initial_radial_distances = {}
        for leg in self.leg_names:
            pos = self.base_feet_pos_body[leg]
            self.initial_radial_distances[leg] = np.sqrt(pos[0]**2 + pos[1]**2)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Coil motion parameters
        self.total_yaw_rotation = 2 * np.pi  # 360 degrees per cycle
        self.base_rise_height = 0.15  # Total vertical rise during coil
        
        # Retraction schedule (fraction of initial radial distance to maintain)
        # Phase 0.0-0.25: 100% -> 100%
        # Phase 0.25-0.5: 100% -> 70%
        # Phase 0.5-0.75: 70% -> 40%
        # Phase 0.75-1.0: 40% -> 20% (near centerline)
        self.min_radial_fraction = 0.2  # Closest approach to centerline

    def update_base_motion(self, phase, dt):
        """
        Update base state: continuous yaw rotation with progressive increase,
        vertical rise, and zero horizontal translation.
        """
        # Yaw rate increases progressively through the cycle
        # Designed to accumulate exactly 360 degrees over full cycle
        # Using smooth acceleration profile
        if phase < 0.25:
            # Initial coil: moderate yaw rate
            yaw_rate_factor = 0.8
        elif phase < 0.5:
            # Progressive coil: maintain yaw rate
            yaw_rate_factor = 0.9
        elif phase < 0.75:
            # Tight coil: increased yaw rate
            yaw_rate_factor = 1.1
        else:
            # Maximum coil: peak yaw rate
            yaw_rate_factor = 1.2
        
        # Base yaw rate to accumulate total_yaw_rotation over one cycle
        base_yaw_rate = self.total_yaw_rotation * self.freq
        yaw_rate = base_yaw_rate * yaw_rate_factor
        
        # Vertical velocity increases as legs retract
        # More rise in later phases when stance is tighter
        if phase < 0.25:
            vz = self.base_rise_height * self.freq * 0.3
        elif phase < 0.5:
            vz = self.base_rise_height * self.freq * 0.8
        elif phase < 0.75:
            vz = self.base_rise_height * self.freq * 1.2
        else:
            vz = self.base_rise_height * self.freq * 1.5
        
        # No horizontal translation - pure in-place rotation
        self.vel_world = np.array([0.0, 0.0, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate base pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with progressive radial retraction.
        
        All feet retract symmetrically toward body centerline while maintaining
        ground contact (z = constant relative to initial position).
        """
        initial_pos = self.base_feet_pos_body[leg_name].copy()
        initial_radial = self.initial_radial_distances[leg_name]
        
        # Compute target radial fraction based on phase
        if phase < 0.25:
            # Initial coil: no retraction yet, full extension
            radial_fraction = 1.0
        elif phase < 0.5:
            # Progressive coil: retract from 100% to 70%
            local_progress = (phase - 0.25) / 0.25
            radial_fraction = 1.0 - 0.3 * local_progress
        elif phase < 0.75:
            # Tight coil: retract from 70% to 40%
            local_progress = (phase - 0.5) / 0.25
            radial_fraction = 0.7 - 0.3 * local_progress
        else:
            # Maximum coil: retract from 40% to 20%
            local_progress = (phase - 0.75) / 0.25
            radial_fraction = 0.4 - 0.2 * local_progress
        
        # Smooth the retraction with ease-in-out within each sub-phase
        radial_fraction = max(self.min_radial_fraction, radial_fraction)
        
        # Compute new foot position by scaling radial distance
        if initial_radial > 1e-6:  # Avoid division by zero
            # Scale x and y components to achieve target radial distance
            scale = radial_fraction
            foot_pos = initial_pos.copy()
            foot_pos[0] = initial_pos[0] * scale
            foot_pos[1] = initial_pos[1] * scale
            # Z coordinate remains at ground level (relative to initial stance)
            foot_pos[2] = initial_pos[2]
        else:
            # Leg already at centerline, no change needed
            foot_pos = initial_pos.copy()
        
        return foot_pos