from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_LATERAL_COIL_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Lateral coil spin: in-place yaw rotation with progressive leg retraction.
    
    - All four legs remain in continuous ground contact (stance)
    - Legs retract radially inward toward body centerline over phase [0,1]
    - Base rises to maintain stability as stance narrows
    - Yaw rate increases progressively through the cycle
    - One full cycle completes ~360° rotation with maximum coil at phase=1.0
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (body frame, initial wide stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compute radial direction for each leg in horizontal plane
        self.leg_radial_dirs = {}
        for leg_name, pos in self.base_feet_pos_body.items():
            horizontal_pos = np.array([pos[0], pos[1], 0.0])
            radial_dist = np.linalg.norm(horizontal_pos)
            if radial_dist > 1e-6:
                self.leg_radial_dirs[leg_name] = horizontal_pos / radial_dist
            else:
                self.leg_radial_dirs[leg_name] = np.array([1.0, 0.0, 0.0])
        
        # Retraction parameters: percentage of radial distance to retract at phase=1.0
        self.max_retraction_ratio = 0.75  # Retract 75% toward centerline at peak coil
        
        # Base rise parameters
        self.max_base_rise = 0.15  # Maximum base height increase (m)
        
        # Yaw rotation parameters
        self.yaw_rate_initial = 2.0  # rad/s at phase 0
        self.yaw_rate_final = 4.0    # rad/s at phase 1 (doubled)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with progressive yaw rotation and upward rise.
        
        Yaw rate increases linearly from initial to final over the phase cycle.
        Upward velocity increases quadratically to create smooth base rise.
        No lateral (x, y) translation.
        """
        # Progressive yaw rate: linear interpolation from initial to final
        yaw_rate = self.yaw_rate_initial + (self.yaw_rate_final - self.yaw_rate_initial) * phase
        
        # Upward velocity: quadratic profile for smooth acceleration
        # vz peaks mid-cycle and integrates to total rise by phase=1.0
        # Using smooth polynomial: vz = 4 * max_rise * freq * phase * (1 - phase)
        # This integrates to max_rise * (2*phase^2 - (2/3)*phase^3) over [0,1]
        # Adjusted to reach max_base_rise at phase=1.0
        vz = 6.0 * self.max_base_rise * self.freq * phase * (1.0 - phase / 1.5)
        
        # Set velocity commands in world frame
        self.vel_world = np.array([0.0, 0.0, vz])
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
        Compute foot position in body frame with radial retraction.
        
        Each foot moves radially inward toward the body centerline in the
        horizontal plane while extending vertically to maintain ground contact
        as the base rises.
        
        Retraction profile:
        - phase 0.0-0.25: 15% retraction
        - phase 0.25-0.5: 30% retraction
        - phase 0.5-0.75: 60% retraction
        - phase 0.75-1.0: 75% retraction (maximum coil)
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Compute current retraction ratio using smooth progression
        # Using cubic easing for smooth acceleration into tight coil
        if phase < 0.25:
            retraction_ratio = 0.15 * (phase / 0.25)
        elif phase < 0.5:
            retraction_ratio = 0.15 + 0.15 * ((phase - 0.25) / 0.25)
        elif phase < 0.75:
            retraction_ratio = 0.30 + 0.30 * ((phase - 0.5) / 0.25)
        else:
            retraction_ratio = 0.60 + 0.15 * ((phase - 0.75) / 0.25)
        
        # Horizontal retraction: move radially inward
        horizontal_base = np.array([base_pos[0], base_pos[1]])
        radial_distance = np.linalg.norm(horizontal_base)
        
        if radial_distance > 1e-6:
            retraction_amount = radial_distance * retraction_ratio
            foot_horizontal = horizontal_base * (1.0 - retraction_ratio)
        else:
            foot_horizontal = horizontal_base
        
        # Vertical extension: adjust to maintain ground contact as base rises
        # As base rises, foot z-position (body frame) must become more negative
        # to keep foot on ground (z=0 in world frame)
        base_rise_current = self.max_base_rise * (2.0 * phase**2 - (2.0/3.0) * phase**3)
        
        # Additional vertical extension needed in body frame
        vertical_offset = -base_rise_current
        
        foot_pos = np.array([
            foot_horizontal[0],
            foot_horizontal[1],
            base_pos[2] + vertical_offset
        ])
        
        return foot_pos