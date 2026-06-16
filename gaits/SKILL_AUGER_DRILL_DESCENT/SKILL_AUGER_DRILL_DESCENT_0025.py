from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_AUGER_DRILL_DESCENT_MotionGenerator(BaseMotionGenerator):
    """
    Auger drill descent motion with continuous yaw rotation and vertical descent.
    
    - Base rotates continuously around vertical axis (yaw) at constant rate
    - Base descends vertically at constant negative z-velocity
    - All four legs remain extended in stance, maintaining ground contact
    - Legs trace helical paths in world frame; in body frame they adjust downward
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Extended stance configuration - increase radial extension for drill effect
        # Extend feet outward by 1.3x to create wider drill-bit appearance
        self.extension_factor = 1.3
        for leg_name in self.leg_names:
            foot = self.base_feet_pos_body[leg_name]
            # Extend radially in x-y plane
            radial_xy = np.sqrt(foot[0]**2 + foot[1]**2)
            if radial_xy > 1e-6:
                scale = self.extension_factor
                foot[0] *= scale
                foot[1] *= scale
        
        # Descent parameters
        self.total_descent = 0.3  # Total vertical descent over one cycle (meters)
        self.descent_velocity = -self.total_descent * self.freq  # Constant negative z-velocity
        
        # Rotation parameters - 3 full rotations per cycle for clear drill effect
        self.rotations_per_cycle = 3.0
        self.yaw_rate = 2.0 * np.pi * self.rotations_per_cycle * self.freq  # radians per second
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (constant throughout)
        self.vel_world = np.array([0.0, 0.0, self.descent_velocity])
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate])

    def update_base_motion(self, phase, dt):
        """
        Update base with constant yaw rotation and constant downward velocity.
        No variation with phase - uniform drilling motion throughout.
        """
        # Constant velocities throughout entire motion
        self.vel_world = np.array([0.0, 0.0, self.descent_velocity])
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate])
        
        # Integrate pose in world frame
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame.
        
        Feet maintain extended stance configuration throughout.
        In body frame, feet adjust downward to match body descent rate,
        maintaining ground contact as body spirals downward.
        """
        # Start from extended base position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Adjust z-coordinate downward to compensate for body descent
        # This maintains ground contact in world frame while body descends
        # phase = 0: no adjustment, phase = 1: full descent adjustment
        foot[2] -= self.total_descent * phase
        
        # All four legs maintain symmetric extended stance
        # No swing phase - continuous ground contact
        
        return foot