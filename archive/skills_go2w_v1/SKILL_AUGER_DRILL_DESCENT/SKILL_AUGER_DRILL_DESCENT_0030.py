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
        
        # Moderate radial extension for drill effect
        self.extension_factor = 1.06
        for leg_name in self.leg_names:
            foot = self.base_feet_pos_body[leg_name]
            radial_xy = np.sqrt(foot[0]**2 + foot[1]**2)
            if radial_xy > 1e-6:
                foot[0] *= self.extension_factor
                foot[1] *= self.extension_factor
            # Minimal baseline elevation for extension geometry compensation only
            foot[2] += 0.028
        
        # Descent parameters
        self.total_descent = 0.14
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
        UPWARD relative to the body frame. We apply upward compensation based on phase,
        ensuring phase wraps correctly to [0,1] range for cyclic motion.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Ensure phase is in [0, 1] range - handle wrapping for multi-cycle motions
        effective_phase = phase % 1.0
        
        # Apply upward z-adjustment to compensate for body descent
        # Compensation matches descent rate exactly with no additional margin
        upward_compensation = self.total_descent * effective_phase
        foot[2] += upward_compensation
        
        return foot