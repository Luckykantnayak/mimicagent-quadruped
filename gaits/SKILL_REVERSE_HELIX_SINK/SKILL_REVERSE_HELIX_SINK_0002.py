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
    - Progressive height descent (controlled position-based descent)
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
        self.max_descent = 0.18  # Maximum height drop (reduced to maintain safe clearance)
        self.min_safe_height = 0.15  # Absolute minimum height above ground
        
        # Stance spreading parameters
        self.max_radial_spread = 1.15  # Legs spread outward by 15% at maximum compression
        
        # Store world-frame foot positions (ground contact points)
        self.world_foot_positions = {}

    def reset(self, root_pos, root_quat):
        """Reset the motion generator state."""
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0
        self.initial_height = root_pos[2]
        
        # Initialize world-frame foot positions at ground level
        for leg in self.leg_names:
            foot_body = self.base_feet_pos_body[leg].copy()
            foot_world = quat_rotate(self.root_quat, foot_body) + self.root_pos
            # Set ground contact at z=0
            foot_world[2] = 0.0
            self.world_foot_positions[leg] = foot_world

    def get_target_height(self, phase):
        """
        Compute target base height as explicit function of phase.
        Position-based control to prevent uncontrolled descent.
        """
        # Phase-dependent descent profile
        if phase < 0.25:
            # Initiation: gradual descent start (0 to 20% of max)
            progress = phase / 0.25
            descent_fraction = 0.2 * progress
        elif phase < 0.5:
            # Acceleration: descent to 60% of max
            progress = (phase - 0.25) / 0.25
            descent_fraction = 0.2 + 0.4 * progress
        elif phase < 0.75:
            # Deep descent: approach 90% of max
            progress = (phase - 0.5) / 0.25
            descent_fraction = 0.6 + 0.3 * progress
        else:
            # Completion: reach 100% of max descent
            progress = (phase - 0.75) / 0.25
            descent_fraction = 0.9 + 0.1 * progress
        
        target_descent = self.max_descent * descent_fraction
        target_height = self.initial_height - target_descent
        
        # Enforce absolute minimum safe height
        target_height = max(target_height, self.min_safe_height)
        
        return target_height

    def update_base_motion(self, phase, dt):
        """
        Update base pose using:
        - Constant backward velocity (negative vx)
        - Constant counter-clockwise yaw rate
        - Position-controlled height descent with velocity smoothing
        """
        
        # Constant backward velocity
        vx = self.vx_backward
        
        # Constant counter-clockwise yaw rate
        yaw_rate = self.yaw_rate
        
        # Compute target height for current phase
        target_height = self.get_target_height(phase)
        
        # Compute smooth velocity to reach target height
        height_error = target_height - self.root_pos[2]
        # Proportional velocity control with reasonable gain
        vz = height_error / (dt * 5.0)  # Smooth approach over ~5 timesteps
        # Clamp velocity to prevent abrupt changes
        vz = np.clip(vz, -0.4, 0.4)
        
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
        Compute foot position in body frame maintaining world-frame ground contact.
        
        Strategy:
        1. Maintain world-frame foot position at ground level (z=0)
        2. Apply radial spreading in world frame for stability
        3. Transform back to body frame accounting for current orientation
        """
        
        # Get base world-frame foot position (at ground)
        foot_world = self.world_foot_positions[leg_name].copy()
        
        # Compute spreading factor based on phase (synchronized with descent)
        if phase < 0.25:
            compression = phase / 0.25 * 0.2
        elif phase < 0.5:
            compression = 0.2 + (phase - 0.25) / 0.25 * 0.4
        elif phase < 0.75:
            compression = 0.6 + (phase - 0.5) / 0.25 * 0.3
        else:
            compression = 0.9 + (phase - 0.75) / 0.25 * 0.1
        
        compression = min(compression, 1.0)
        
        # Apply radial spreading in world frame (horizontal plane only)
        radial_spread_factor = 1.0 + (self.max_radial_spread - 1.0) * compression
        
        # Compute center of base in world frame (horizontal only)
        base_xy_world = self.root_pos[:2]
        
        # Spread foot outward from base center in horizontal plane
        foot_xy_offset = foot_world[:2] - base_xy_world
        foot_xy_dist = np.linalg.norm(foot_xy_offset)
        if foot_xy_dist > 1e-6:
            spread_direction = foot_xy_offset / foot_xy_dist
            foot_world[:2] = base_xy_world + spread_direction * foot_xy_dist * radial_spread_factor
        
        # Maintain ground contact
        foot_world[2] = 0.0
        
        # Transform world-frame foot position to body frame
        # foot_world = quat_rotate(root_quat, foot_body) + root_pos
        # => foot_body = quat_rotate(quat_inverse(root_quat), foot_world - root_pos)
        quat_inv = quat_inverse(self.root_quat)
        foot_body = quat_rotate(quat_inv, foot_world - self.root_pos)
        
        return foot_body