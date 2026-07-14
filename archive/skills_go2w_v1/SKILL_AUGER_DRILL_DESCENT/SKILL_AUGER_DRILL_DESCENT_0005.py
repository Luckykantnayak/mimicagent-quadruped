from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_AUGER_DRILL_DESCENT_MotionGenerator(BaseMotionGenerator):
    """
    Auger drill descent skill.
    
    Robot descends vertically while spinning rapidly about its yaw axis.
    All four legs maintain contact and trace helical paths in world frame
    while remaining approximately stationary in body frame.
    
    Base motion:
    - Constant downward velocity (vz < 0)
    - Constant positive yaw rate (multiple full rotations per cycle)
    - Zero horizontal velocity (vx = vy = 0)
    - Zero roll/pitch rates
    
    Leg motion:
    - All four legs extended and in continuous stance
    - Body-frame positions approximately constant
    - World-frame trajectories are helical (from body rotation + descent)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions - extended configuration for stability during rotation
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Extend legs radially for wider support polygon during rapid rotation
        # Scale radial distance by 1.2 to increase stability
        for leg_name in self.leg_names:
            foot = self.base_feet_pos_body[leg_name]
            radial_xy = np.sqrt(foot[0]**2 + foot[1]**2)
            if radial_xy > 0:
                scale_factor = 1.2
                foot[0] *= scale_factor
                foot[1] *= scale_factor
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Drill motion parameters
        # Descent rate: constant downward velocity in m/s
        self.descent_rate = -0.15  # negative for downward motion
        
        # Yaw rate: target 3 full rotations per cycle (6π radians per cycle)
        # If freq = 1.0 Hz, cycle period T = 1.0 second
        # yaw_rate = 6π rad/s achieves 3 full rotations in 1 second
        self.yaw_rate = 6.0 * np.pi  # rad/s (3 full rotations per cycle)
        
        # Zero horizontal velocities and roll/pitch rates
        self.vel_world = np.array([0.0, 0.0, self.descent_rate])
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate])

    def update_base_motion(self, phase, dt):
        """
        Update base with constant downward velocity and constant yaw rate.
        
        The drill motion consists of:
        - Constant vz (descent_rate < 0)
        - Constant yaw_rate (positive, rapid rotation)
        - All other velocity components zero
        """
        # Constant velocity commands - no phase dependence
        self.vel_world = np.array([0.0, 0.0, self.descent_rate])
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
        Compute foot position in body frame.
        
        All legs remain in extended stance configuration throughout the cycle.
        Body-frame positions are approximately constant - the helical world-frame
        trajectories emerge naturally from the body's rotation and descent.
        
        Minor vertical adjustments maintain ground contact during descent.
        """
        # Start from extended base position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # All legs maintain constant body-frame positions
        # No swing phase - continuous stance for all legs
        # The helical motion in world frame comes from body rotation + descent
        
        # Optional: small vertical adjustment to maintain contact during descent
        # This can help compensate for any body pitch/roll oscillations
        # For pure kinematic drill motion, base position is sufficient
        
        return foot