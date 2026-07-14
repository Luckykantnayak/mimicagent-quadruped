from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_AUGER_DRILL_DESCENT_MotionGenerator(BaseMotionGenerator):
    """
    Auger drill descent motion: continuous rapid yaw rotation combined with uniform
    vertical descent. All four legs trace helical paths in body frame, creating a
    drill-bit-like visual appearance.
    
    - Base: constant yaw rate (multiple rotations per cycle) + constant downward velocity
    - Legs: helical trajectories in body frame (circular XY motion + compensated Z)
    - Contact: all four legs maintain continuous ground contact throughout
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0  # One full phase cycle per second
        
        # Base foot positions (body frame reference)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Helical trajectory parameters - reduced radius to respect workspace limits
        self.helix_radius = 0.10  # Reduced from 0.20 to keep within kinematic workspace
        self.helix_rotations = 1.0  # Number of complete XY rotations per phase cycle
        
        # Angular offsets for symmetric four-leg distribution (90° spacing)
        self.angular_offsets = {}
        for i, leg in enumerate(leg_names):
            if leg.startswith('FL'):
                self.angular_offsets[leg] = 0.0
            elif leg.startswith('FR'):
                self.angular_offsets[leg] = 0.5 * np.pi
            elif leg.startswith('RL'):
                self.angular_offsets[leg] = np.pi
            elif leg.startswith('RR'):
                self.angular_offsets[leg] = 1.5 * np.pi
        
        # Base motion parameters
        self.yaw_rate = 4.0 * np.pi  # Constant yaw rate (2 full rotations per cycle)
        self.descent_velocity = -0.15  # Constant downward velocity (m/s)
        
        # Z compensation: feet must ascend in body frame to compensate for base descent
        # This prevents ground penetration while maintaining drilling visual
        self.body_frame_z_compensation = 0.15  # Matches base descent magnitude
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update_base_motion(self, phase, dt):
        """
        Apply constant yaw rotation rate and constant downward velocity throughout
        the entire phase cycle to create the drilling descent motion.
        """
        # Constant downward velocity in world frame
        self.vel_world = np.array([0.0, 0.0, self.descent_velocity])
        
        # Constant positive yaw rate (rotation about vertical axis)
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate])
        
        # Integrate pose using world-frame velocities
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute helical trajectory for each leg in body frame.
        
        Each leg traces a helix:
        - XY: circular path with radius helix_radius, angle advances with phase
        - Z: compensated to counteract base descent and prevent ground penetration
        - Angular offset creates symmetric drill-bit distribution
        """
        # Get base position for this leg
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Compute helical trajectory parameters
        # Angular position advances continuously with phase (multiple rotations possible)
        angular_progress = 2.0 * np.pi * self.helix_rotations * phase
        angle = angular_progress + self.angular_offsets[leg_name]
        
        # Helical position in body frame
        foot = np.zeros(3)
        
        # XY: circular motion at constant radius (reduced for workspace compatibility)
        foot[0] = self.helix_radius * np.cos(angle)
        foot[1] = self.helix_radius * np.sin(angle)
        
        # Z: compensate for base descent to prevent ground penetration
        # Feet ascend in body frame as base descends in world frame
        # This keeps feet at consistent world-frame height while body descends
        foot[2] = base_pos[2] + self.body_frame_z_compensation * phase
        
        return foot