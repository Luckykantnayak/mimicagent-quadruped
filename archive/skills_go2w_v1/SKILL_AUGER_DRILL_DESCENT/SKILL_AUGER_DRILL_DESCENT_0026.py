from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_AUGER_DRILL_DESCENT_MotionGenerator(BaseMotionGenerator):
    """
    Auger drill descent motion with continuous yaw rotation and vertical descent.
    
    - Base rotates continuously around vertical axis (yaw) at constant rate
    - Base descends vertically at constant negative z-velocity
    - All four legs remain extended in stance, maintaining ground contact
    - Legs trace helical paths in world frame; in body frame they adjust upward to compensate for descent
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Moderate radial extension for drill effect without exceeding kinematic limits
        self.extension_factor = 1.1
        for leg_name in self.leg_names:
            foot = self.base_feet_pos_body[leg_name]
            radial_xy = np.sqrt(foot[0]**2 + foot[1]**2)
            if radial_xy > 1e-6:
                foot[0] *= self.extension_factor
                foot[1] *= self.extension_factor
        
        # Descent parameters - reduced for safer operation
        self.total_descent = 0.18
        self.descent_velocity = -self.total_descent * self.freq
        
        # Rotation parameters - 3 full rotations per cycle
        self.rotations_per_cycle = 3.0
        self.yaw_rate = 2.0 * np.pi * self.rotations_per_cycle * self.freq
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.array([0.0, 0.0, self.descent_velocity])
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate])

    def update_base_motion(self, phase, dt):
        """
        Update base with constant yaw rotation and constant downward velocity.
        """
        self.vel_world = np.array([0.0, 0.0, self.descent_velocity])
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate])
        
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
        
        As the body descends in world frame, ground-fixed foot positions appear to move
        UPWARD relative to the body frame. To maintain constant world-frame ground contact,
        we adjust body-frame foot z-coordinate progressively upward.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Adjust z-coordinate UPWARD to compensate for body descent
        # As body moves down in world frame, ground appears to move up in body frame
        foot[2] += self.total_descent * phase
        
        return foot