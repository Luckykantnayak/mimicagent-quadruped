from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_AUGER_DRILL_DESCENT_MotionGenerator(BaseMotionGenerator):
    """
    Auger drill descent skill: robot descends vertically while spinning rapidly about yaw axis.
    
    - All four legs remain in continuous ground contact throughout the motion
    - Base descends at constant downward velocity (negative vz)
    - Base rotates at constant yaw rate (multiple full rotations per cycle)
    - Legs maintain extended radial positions in body frame with proper height compensation
    - World-frame foot trajectories trace helical paths due to integrated base motion
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (extended radially for wide support polygon)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Extend legs radially to maximize support polygon during rotation
        # Apply horizontal radial extension with compensating vertical adjustment
        self.radial_extension = 1.15  # Moderate extension to stay safely within workspace
        for leg in self.leg_names:
            # Store original values
            orig_x = self.base_feet_pos_body[leg][0]
            orig_y = self.base_feet_pos_body[leg][1]
            orig_z = self.base_feet_pos_body[leg][2]
            
            # Apply horizontal extension
            self.base_feet_pos_body[leg][0] = orig_x * self.radial_extension
            self.base_feet_pos_body[leg][1] = orig_y * self.radial_extension
            
            # Compensate vertical position: when legs extend horizontally,
            # they cannot reach as far down. Raise foot z to maintain reachability
            # Increased compensation to 22% to provide more initial clearance
            z_compensation = 0.22 * abs(orig_z)
            self.base_feet_pos_body[leg][2] = orig_z + z_compensation
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Drilling motion parameters - moderate values to maintain stability
        self.descent_rate = -0.09  # Reduced descent rate for extended safe operation
        self.yaw_rate = 3.0 * 2.0 * np.pi  # 3 full rotations per phase cycle
        
        # Velocity commands (world frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base with constant downward velocity and constant yaw rotation.
        This produces a helical descent trajectory.
        """
        # Constant downward velocity (purely vertical descent, no lateral drift)
        self.vel_world = np.array([0.0, 0.0, self.descent_rate])
        
        # Constant yaw rate (continuous rotation about vertical axis)
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
        All legs maintain continuous ground contact in extended stance configuration.
        Foot positions remain relatively static in body frame throughout the cycle.
        The helical world-frame trajectories emerge from base motion integration.
        
        Strong dynamic adjustment compensates for descent to maintain ground contact feasibility.
        """
        # Start with base extended foot position (already compensated for radial extension)
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Dynamic vertical adjustment: as body descends, raise body-frame foot z significantly
        # to maintain reachability and prevent ground penetration as geometry changes
        # Increased compensation factor from 0.1 to 0.6 to properly counteract cumulative descent
        # This creates body-frame upward drift that compensates for world-frame base descent,
        # keeping world-frame foot positions approximately stationary (grounded) during stance
        descent_compensation = -self.descent_rate * self.t * 0.6
        foot[2] = self.base_feet_pos_body[leg_name][2] + descent_compensation
        
        return foot