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
        self.initial_angles = {}
        for leg in self.leg_names:
            pos = self.base_feet_pos_body[leg]
            self.initial_radial_distances[leg] = np.sqrt(pos[0]**2 + pos[1]**2)
            self.initial_angles[leg] = np.arctan2(pos[1], pos[0])
        
        # Store initial ground contact z-position in body frame
        self.ground_z_body = {}
        for leg in self.leg_names:
            self.ground_z_body[leg] = self.base_feet_pos_body[leg][2]
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Coil motion parameters
        self.total_yaw_rotation = 2 * np.pi  # 360 degrees per cycle
        self.base_rise_height = 0.08  # Reduced total vertical rise during coil
        
        # Retraction schedule (fraction of initial radial distance to maintain)
        self.min_radial_fraction = 0.3  # Closest approach to centerline (was 0.2, increased for safety)
        
        # Track cumulative base vertical displacement
        self.cumulative_base_rise = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base state: continuous yaw rotation with progressive increase,
        vertical rise, and zero horizontal translation.
        """
        # Yaw rate increases progressively through the cycle
        # Using smooth profile to accumulate exactly 360 degrees over full cycle
        if phase < 0.25:
            yaw_rate_factor = 0.85
        elif phase < 0.5:
            yaw_rate_factor = 0.95
        elif phase < 0.75:
            yaw_rate_factor = 1.05
        else:
            yaw_rate_factor = 1.15
        
        # Base yaw rate to accumulate total_yaw_rotation over one cycle
        base_yaw_rate = self.total_yaw_rotation * self.freq
        yaw_rate = base_yaw_rate * yaw_rate_factor
        
        # Vertical velocity increases as legs retract, but with smoother profile
        # Total vertical displacement should equal base_rise_height over full cycle
        if phase < 0.25:
            vz_factor = 0.4
        elif phase < 0.5:
            vz_factor = 0.8
        elif phase < 0.75:
            vz_factor = 1.1
        else:
            vz_factor = 1.3
        
        # Normalize vertical velocity to achieve target rise height
        vz = self.base_rise_height * self.freq * vz_factor
        
        # Track cumulative rise for foot compensation
        self.cumulative_base_rise += vz * dt
        
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
        Compute foot position in body frame with progressive radial retraction
        and vertical compensation to maintain ground contact.
        
        All feet retract symmetrically toward body centerline while foot z-coordinate
        in body frame decreases to compensate for rising base, keeping world-frame
        z-position at ground level.
        """
        initial_pos = self.base_feet_pos_body[leg_name].copy()
        initial_radial = self.initial_radial_distances[leg_name]
        
        # Compute target radial fraction based on phase with smooth transitions
        if phase < 0.25:
            # Initial coil: minimal retraction, mostly at full extension
            t = phase / 0.25
            t_smooth = 3*t**2 - 2*t**3  # Smooth ease
            radial_fraction = 1.0 - 0.05 * t_smooth
        elif phase < 0.5:
            # Progressive coil: retract from 95% to 70%
            t = (phase - 0.25) / 0.25
            t_smooth = 3*t**2 - 2*t**3
            radial_fraction = 0.95 - 0.25 * t_smooth
        elif phase < 0.75:
            # Tight coil: retract from 70% to 45%
            t = (phase - 0.5) / 0.25
            t_smooth = 3*t**2 - 2*t**3
            radial_fraction = 0.70 - 0.25 * t_smooth
        else:
            # Maximum coil: retract from 45% to 30%
            t = (phase - 0.75) / 0.25
            t_smooth = 3*t**2 - 2*t**3
            radial_fraction = 0.45 - 0.15 * t_smooth
        
        radial_fraction = max(self.min_radial_fraction, radial_fraction)
        
        # Compute new foot position by scaling radial distance
        if initial_radial > 1e-6:
            # Scale x and y components to achieve target radial distance
            scale = radial_fraction
            foot_pos = initial_pos.copy()
            foot_pos[0] = initial_pos[0] * scale
            foot_pos[1] = initial_pos[1] * scale
        else:
            foot_pos = initial_pos.copy()
        
        # CRITICAL: Adjust z-coordinate in body frame to compensate for base rise
        # As base rises in world frame, foot must descend in body frame to maintain ground contact
        # Compute expected base rise at this phase
        phase_base_rise = self.compute_expected_base_rise_at_phase(phase)
        
        # Foot z in body frame = initial ground z - cumulative base rise
        # This keeps world-frame foot z approximately constant at ground level
        foot_pos[2] = self.ground_z_body[leg_name] - phase_base_rise
        
        return foot_pos

    def compute_expected_base_rise_at_phase(self, phase):
        """
        Compute expected cumulative base vertical displacement at given phase.
        This is used to adjust foot z-coordinates in body frame to maintain ground contact.
        """
        # Integrate the vertical velocity profile numerically over phase
        # Using the same profile as update_base_motion
        total_rise = 0.0
        num_steps = 100
        for i in range(num_steps):
            p = phase * i / num_steps
            if p < 0.25:
                vz_factor = 0.4
            elif p < 0.5:
                vz_factor = 0.8
            elif p < 0.75:
                vz_factor = 1.1
            else:
                vz_factor = 1.3
            
            # Accumulate rise contribution from this phase segment
            # vz = base_rise_height * freq * vz_factor
            # displacement = vz * dt, where dt = (1 cycle / freq) / num_steps
            phase_step = phase / num_steps
            time_step = phase_step / self.freq
            vz = self.base_rise_height * self.freq * vz_factor
            total_rise += vz * time_step
        
        return total_rise