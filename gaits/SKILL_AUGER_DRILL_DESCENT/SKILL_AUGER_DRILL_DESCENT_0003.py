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
        
        # Helical trajectory parameters - small radius as offset from natural stance
        self.helix_radius = 0.04  # Small offset to create visible spiraling without workspace violation
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
        self.descent_rate = 0.15  # Total descent over one cycle (m)
        
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
        descent_velocity = -self.descent_rate * self.freq
        self.vel_world = np.array([0.0, 0.0, descent_velocity])
        
        # Constant positive yaw rate (rotation about vertical axis)
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate * self.freq])
        
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
        
        Each leg traces a helix around its natural stance position:
        - XY: circular offset from base position with small radius
        - Z: compensated to maintain world-frame ground contact as base descends
        - Angular offset creates symmetric drill-bit distribution
        """
        # Get base position for this leg (natural stance position in body frame)
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Compute helical trajectory parameters
        # Angular position advances continuously with phase
        angular_progress = 2.0 * np.pi * self.helix_rotations * phase
        angle = angular_progress + self.angular_offsets[leg_name]
        
        # Smooth phase transition using cosine envelope to reduce jerk at boundaries
        phase_smooth = 0.5 * (1.0 - np.cos(np.pi * phase))
        
        # Helical offset from base stance position
        helix_offset_x = self.helix_radius * np.cos(angle)
        helix_offset_y = self.helix_radius * np.sin(angle)
        
        # XY: Add helical offset to natural stance position (not absolute positioning)
        foot_x = base_pos[0] + helix_offset_x
        foot_y = base_pos[1] + helix_offset_y
        
        # Z: Compensate for base descent to maintain ground contact
        # As base descends in world frame, feet must rise in body frame by equal amount
        # to keep world-frame Z position constant (maintaining ground contact)
        z_compensation = self.descent_rate * phase_smooth
        foot_z = base_pos[2] + z_compensation
        
        foot = np.array([foot_x, foot_y, foot_z])
        
        return foot