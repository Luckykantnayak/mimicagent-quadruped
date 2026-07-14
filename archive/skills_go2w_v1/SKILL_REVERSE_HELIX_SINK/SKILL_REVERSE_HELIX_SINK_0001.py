from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERSE_HELIX_SINK_MotionGenerator(BaseMotionGenerator):
    """
    Reverse helical descent motion generator.
    
    The robot executes backward translation combined with continuous counter-clockwise
    yaw rotation while progressively lowering its base height. Over one full phase cycle,
    the robot completes one 360° helical loop while sinking toward the ground.
    
    Key characteristics:
    - Constant backward velocity (negative vx)
    - Constant yaw rate (counter-clockwise, completing 360° per cycle)
    - Progressive height descent (vz negative, tapering to zero)
    - All four feet remain in contact throughout
    - Legs compress and spread outward for stability during descent
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Lower frequency for controlled helical descent
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Store initial radial distances for stance spreading
        self.initial_radial_dist = {}
        for leg in self.leg_names:
            pos = self.base_feet_pos_body[leg]
            self.initial_radial_dist[leg] = np.sqrt(pos[0]**2 + pos[1]**2)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.vx_backward = -0.5  # Constant backward velocity
        self.yaw_rate = 2 * np.pi * self.freq  # 360° per cycle (2π radians)
        
        # Height descent parameters
        self.initial_height = 0.0  # Will be set relative to initial base height
        self.max_descent = 0.25  # Maximum height drop
        
        # Stance spreading parameters
        self.max_radial_spread = 1.15  # Legs spread outward by 15% at maximum compression
        
        # Leg compression parameters
        self.max_z_compression = 0.25  # Maximum downward foot shift in body frame

    def reset(self, root_pos, root_quat):
        """Reset the motion generator state."""
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0
        self.initial_height = root_pos[2]

    def update_base_motion(self, phase, dt):
        """
        Update base pose using:
        - Constant backward velocity (negative vx)
        - Constant counter-clockwise yaw rate
        - Height-dependent downward velocity (vz)
        
        Phase-dependent velocity profile:
        - [0.0, 0.25]: Initiate descent
        - [0.25, 0.5]: Peak descent rate
        - [0.5, 0.75]: Moderate descent approaching minimum
        - [0.75, 1.0]: Taper to zero, stabilize at minimum height
        """
        
        # Constant backward velocity
        vx = self.vx_backward
        
        # Constant counter-clockwise yaw rate
        yaw_rate = self.yaw_rate
        
        # Phase-dependent downward velocity profile
        if phase < 0.25:
            # Initiation: gradual descent start
            progress = phase / 0.25
            vz = -self.max_descent * 1.5 * progress
        elif phase < 0.5:
            # Acceleration: peak descent rate
            vz = -self.max_descent * 1.5
        elif phase < 0.75:
            # Deep descent: moderate rate approaching minimum
            progress = (phase - 0.5) / 0.25
            vz = -self.max_descent * 1.5 * (1.0 - 0.6 * progress)
        else:
            # Completion: taper to zero
            progress = (phase - 0.75) / 0.25
            vz = -self.max_descent * 1.5 * (0.4 * (1.0 - progress))
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, 0.0, vz])
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
        Compute foot position in body frame with:
        1. Progressive leg compression (downward z shift)
        2. Radial spreading for wider support polygon
        3. All feet remain in contact (stance throughout)
        
        The body frame rotates with the base, so foot positions in body frame
        must be adjusted to maintain world-frame ground contact as the base
        yaws and descends.
        """
        
        # Start with base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute compression factor based on phase
        # Progressive compression matching base height descent
        if phase < 0.25:
            compression = phase / 0.25 * 0.3
        elif phase < 0.5:
            compression = 0.3 + (phase - 0.25) / 0.25 * 0.4
        elif phase < 0.75:
            compression = 0.7 + (phase - 0.5) / 0.25 * 0.25
        else:
            compression = 0.95 + (phase - 0.75) / 0.25 * 0.05
        
        compression = min(compression, 1.0)
        
        # Apply vertical compression (raise feet in body frame as base descends)
        # This maintains ground contact while base lowers
        foot[2] += self.max_z_compression * compression
        
        # Compute radial spreading for stability
        radial_spread_factor = 1.0 + (self.max_radial_spread - 1.0) * compression
        
        # Apply radial spreading (outward from body center)
        xy_radial = np.sqrt(foot[0]**2 + foot[1]**2)
        if xy_radial > 1e-6:
            foot[0] *= radial_spread_factor
            foot[1] *= radial_spread_factor
        
        return foot