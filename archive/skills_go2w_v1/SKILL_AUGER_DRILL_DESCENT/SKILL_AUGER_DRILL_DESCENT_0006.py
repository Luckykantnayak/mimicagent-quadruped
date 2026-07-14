from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_AUGER_DRILL_DESCENT_MotionGenerator(BaseMotionGenerator):
    """
    Auger drill descent skill.
    
    Robot descends vertically while spinning rapidly about its yaw axis.
    All four legs maintain contact and trace helical paths in world frame
    while adjusting body-frame z-positions to compensate for base descent.
    
    Base motion:
    - Constant downward velocity (vz < 0)
    - Constant positive yaw rate (multiple full rotations per cycle)
    - Zero horizontal velocity (vx = vy = 0)
    - Zero roll/pitch rates
    
    Leg motion:
    - All four legs extended and in continuous stance
    - Body-frame x,y positions constant (extended radially)
    - Body-frame z positions increase with phase to compensate for descent
    - World-frame trajectories are helical (from body rotation + descent)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions - extended configuration for stability during rotation
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Extend legs radially for wider support polygon during rapid rotation
        # Scale radial distance by 1.15 to increase stability without overextension
        for leg_name in self.leg_names:
            foot = self.base_feet_pos_body[leg_name]
            radial_xy = np.sqrt(foot[0]**2 + foot[1]**2)
            if radial_xy > 0:
                scale_factor = 1.15
                foot[0] *= scale_factor
                foot[1] *= scale_factor
            # Ensure initial z-position is at or slightly above ground
            # Assume initial foot positions are already at ground level
            # Set to small positive value to ensure no initial penetration
            foot[2] = max(foot[2], -0.001)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Drill motion parameters
        # Descent rate: constant downward velocity in m/s
        self.descent_rate = -0.12  # negative for downward motion, reduced for stability
        
        # Yaw rate: target 3 full rotations per cycle (6π radians per cycle)
        # If freq = 1.0 Hz, cycle period T = 1.0 second
        # yaw_rate = 6π rad/s achieves 3 full rotations in 1 second
        self.yaw_rate = 6.0 * np.pi  # rad/s (3 full rotations per cycle)
        
        # Calculate total descent per cycle for foot compensation
        # descent_per_cycle = descent_rate / freq (negative value)
        self.descent_per_cycle = self.descent_rate / self.freq
        
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
        Body-frame x,y positions are constant (extended radially for stability).
        Body-frame z positions increase linearly with phase to compensate for
        base descent, maintaining constant world-frame ground contact height.
        
        The helical world-frame trajectories emerge from body rotation + descent
        while feet maintain ground contact by sliding in spiral patterns.
        """
        # Start from extended base position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Maintain constant x,y in body frame (extended radial position)
        # Adjust z in body frame to compensate for base descent
        # As base descends by descent_per_cycle over phase [0,1],
        # foot z in body frame must increase by -descent_per_cycle to maintain
        # constant world-frame height (ground contact)
        # Since descent_per_cycle is negative, -descent_per_cycle is positive
        z_compensation = -self.descent_per_cycle * phase
        
        foot[2] = foot[2] + z_compensation
        
        return foot