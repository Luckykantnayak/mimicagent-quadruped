from abc import ABC, abstractmethod
from utils.math_utils import *

class BaseMotionGenerator(ABC):
    """
    Abstract base class for all gait controllers.
    Ensures a consistent interface for all skills.
    """

    def __init__(self, base_init_feet_pos, freq=1.0):
        self.freq = freq
        self.base_init_feet_pos = base_init_feet_pos.copy()
        self.base_feet_pos = base_init_feet_pos.copy()

    def reset(self, root_pos, root_quat):
        """
        Reset motion generator state.
        
        Args:
            root_pos: initial root position [x,y,z]
            root_quat: initial root orientation [w,x,y,z]
        """
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0

    def compute_all_foot_positions_body_frame(self, t):
        """
        Compute all foot positions in body frame.
        
        Returns:
            dict {leg_name: [x,y,z] in body frame}
        """
        return {
            leg: self.compute_foot_position_body_frame(leg, t)
            for leg in self.leg_names
        }
    
    def compute_all_foot_positions_world_frame(self, t):
        """
        Compute all foot positions in world frame.
        
        Returns:
            dict {leg_name: [x,y,z] in world frame}
        """
        feet_body = self.compute_all_foot_positions_body_frame(t)
        return {
            leg: body_to_world_position(pos, self.root_pos, self.root_quat)
            for leg, pos in feet_body.items()
        }
    
    def set_velocity_command(self, vx=0.0, vy=0.0, vz=0.0):
        """Set linear velocity command in world frame (m/s)."""
        self.vel_world = np.array([vx, vy, vz])
    
    def set_angular_velocity_command(self, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0):
        """Set angular velocity command in world frame (rad/s)."""
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])

    def step(self, dt):
        """
        Advance motion by one timestep.
        
        Args:
            dt: timestep (s)
            
        Returns:
            dict containing:
                - root_pos: [x, y, z] in world frame
                - root_quat: [w, x, y, z]
                - foot_positions_body: dict {leg_name: [x,y,z]} in body frame
                - foot_positions_world: dict {leg_name: [x,y,z]} in world frame
        """
        
        # Integrate root pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Compute foot positions
        foot_positions_body = self.compute_all_foot_positions_body_frame(self.t)
        foot_positions_world = self.compute_all_foot_positions_world_frame(self.t)
        
        # Advance time
        self.t += dt
        
        return {
            'time': self.t,
            'root_pos': self.root_pos.copy(),
            'root_quat': self.root_quat.copy(),
            'foot_positions_body': foot_positions_body,
            'foot_positions_world': foot_positions_world,
        }
    
    
