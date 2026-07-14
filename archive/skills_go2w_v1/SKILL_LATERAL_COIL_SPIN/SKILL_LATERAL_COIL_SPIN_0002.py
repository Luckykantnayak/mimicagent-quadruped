from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_LATERAL_COIL_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Lateral coil spin: in-place yaw rotation with progressive leg retraction.
    
    - All four legs remain in continuous ground contact (stance)
    - Legs retract radially inward toward body centerline over phase [0,1]
    - Base rises to maintain stability as stance narrows
    - Yaw rate increases progressively through the cycle
    - One full cycle completes ~360° rotation with maximum coil at phase=1.0
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (body frame, initial wide stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Store initial foot positions in world frame (on ground)
        self.initial_feet_world = {}
        for leg_name, pos_body in self.base_feet_pos_body.items():
            # Assume initial base at origin with identity orientation
            self.initial_feet_world[leg_name] = pos_body.copy()
            self.initial_feet_world[leg_name][2] = 0.0  # Enforce ground level
        
        # Compute radial direction for each leg in horizontal plane
        self.leg_radial_dirs = {}
        for leg_name, pos in self.base_feet_pos_body.items():
            horizontal_pos = np.array([pos[0], pos[1], 0.0])
            radial_dist = np.linalg.norm(horizontal_pos)
            if radial_dist > 1e-6:
                self.leg_radial_dirs[leg_name] = horizontal_pos / radial_dist
            else:
                self.leg_radial_dirs[leg_name] = np.array([1.0, 0.0, 0.0])
        
        # Retraction parameters: reduced to respect joint limits
        self.max_retraction_ratio = 0.40  # Retract 40% toward centerline at peak coil
        
        # Base rise parameters: reduced to stay within height envelope
        self.max_base_rise = 0.06  # Maximum base height increase (m)
        
        # Yaw rotation parameters
        self.yaw_rate_initial = 2.0  # rad/s at phase 0
        self.yaw_rate_final = 4.0    # rad/s at phase 1 (doubled)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with progressive yaw rotation and upward rise.
        
        Yaw rate increases linearly from initial to final over the phase cycle.
        Base rises with sinusoidal profile, peaking at mid-cycle and returning to initial height.
        No lateral (x, y) translation.
        """
        # Progressive yaw rate: linear interpolation from initial to final
        yaw_rate = self.yaw_rate_initial + (self.yaw_rate_final - self.yaw_rate_initial) * phase
        
        # Upward velocity: sinusoidal profile for smooth rise and return
        # vz = max_rise * freq * pi * cos(pi * phase)
        # Integrates to max_rise * sin(pi * phase), which peaks at phase=0.5 and returns to 0 at phase=1.0
        vz = self.max_base_rise * self.freq * np.pi * np.cos(np.pi * phase)
        
        # Set velocity commands in world frame
        self.vel_world = np.array([0.0, 0.0, vz])
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
        Compute foot position in body frame with radial retraction while maintaining ground contact.
        
        Each foot moves radially inward toward the body centerline in the horizontal plane.
        Vertical position is computed to maintain z=0 in world frame (continuous ground contact).
        
        Retraction profile follows smooth progression through phases.
        """
        # Get initial foot position in world frame
        initial_world = self.initial_feet_world[leg_name].copy()
        
        # Compute current retraction ratio using smooth cubic progression
        # Gradual acceleration into coil with reduced maximum
        t_norm = phase
        retraction_ratio = self.max_retraction_ratio * (3 * t_norm**2 - 2 * t_norm**3)
        
        # Compute retracted horizontal position in world frame
        horizontal_initial = np.array([initial_world[0], initial_world[1]])
        horizontal_retracted = horizontal_initial * (1.0 - retraction_ratio)
        
        # Foot position in world frame (on ground)
        foot_world = np.array([
            horizontal_retracted[0],
            horizontal_retracted[1],
            0.0  # Always on ground
        ])
        
        # Transform to body frame
        # foot_body = R^T * (foot_world - root_pos)
        # where R is rotation matrix from quaternion
        foot_relative = foot_world - self.root_pos
        
        # Convert quaternion to rotation matrix and apply inverse rotation
        qw, qx, qy, qz = self.root_quat
        
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        # Apply inverse rotation (transpose of R)
        foot_body = R.T @ foot_relative
        
        return foot_body

    def get_state(self, t):
        """
        Main interface: compute full robot state at time t.
        """
        self.t = t
        phase = (self.freq * t) % 1.0
        
        # Compute time step for integration
        dt = 1.0 / 1000.0  # Small fixed timestep for smooth integration
        
        # Update base motion (integrates root_pos and root_quat)
        self.update_base_motion(phase, dt)
        
        # Compute foot positions in body frame
        foot_positions_body = {}
        for leg_name in self.leg_names:
            foot_positions_body[leg_name] = self.compute_foot_position_body_frame(leg_name, phase)
        
        return {
            'root_position': self.root_pos.copy(),
            'root_quaternion': self.root_quat.copy(),
            'foot_positions_body_frame': foot_positions_body,
            'root_linear_velocity': self.vel_world.copy(),
            'root_angular_velocity': self.omega_world.copy()
        }