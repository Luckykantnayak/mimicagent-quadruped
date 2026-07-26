from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Ricochet bounce diagonal gait with alternating diagonal launches.
    
    Motion pattern:
    - Phase 0.0-0.2: Left compression with positive yaw
    - Phase 0.2-0.4: Right-diagonal launch (aerial)
    - Phase 0.4-0.6: Right compression with negative yaw
    - Phase 0.6-0.8: Left-diagonal launch (aerial)
    - Phase 0.8-1.0: Landing preparation
    
    Creates zigzag forward progression through alternating diagonal bounces.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.compression_depth = 0.12  # Vertical compression distance (m)
        self.aerial_height = 0.15  # Peak aerial height (m)
        self.step_length_forward = 0.25  # Forward displacement per cycle
        self.step_length_lateral = 0.15  # Lateral zigzag amplitude
        self.leg_lift_height = 0.10  # Foot clearance during aerial phase
        
        # Velocity parameters
        self.vx_cruise = 0.8  # Forward velocity during aerial phase
        self.vx_compression = 0.3  # Forward velocity during compression
        self.vy_lateral = 0.6  # Lateral velocity magnitude
        self.vz_launch = 1.2  # Upward launch velocity
        self.vz_compression = -0.8  # Downward compression velocity
        
        # Angular velocity parameters
        self.yaw_rate_compression = 2.5  # Yaw rate during compression (rad/s)
        self.max_yaw_angle = np.deg2rad(30)  # Maximum yaw deviation
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity state
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocities based on phase to create ricochet bounce pattern.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.2: Left compression
        if phase < 0.2:
            local_phase = phase / 0.2
            vx = self.vx_compression
            vy = -0.1 * self.vy_lateral  # Slight leftward drift
            vz = self.vz_compression * (1.0 - local_phase)  # Decreasing downward velocity
            yaw_rate = self.yaw_rate_compression  # Positive yaw (left rotation)
        
        # Phase 0.2-0.4: Right-diagonal launch (aerial)
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            vx = self.vx_cruise
            vy = self.vy_lateral  # Rightward velocity
            # Parabolic vertical velocity: starts positive, decreases through aerial phase
            vz = self.vz_launch * (1.0 - 2.0 * local_phase)
            yaw_rate = self.yaw_rate_compression * (1.0 - local_phase)  # Decreasing to zero
        
        # Phase 0.4-0.6: Right compression
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            vx = self.vx_compression
            vy = 0.1 * self.vy_lateral * (1.0 - local_phase)  # Slight rightward then zero
            vz = self.vz_compression * (1.0 - local_phase)  # Decreasing downward velocity
            yaw_rate = -self.yaw_rate_compression  # Negative yaw (right rotation)
        
        # Phase 0.6-0.8: Left-diagonal launch (aerial)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            vx = self.vx_cruise
            vy = -self.vy_lateral  # Leftward velocity
            # Parabolic vertical velocity
            vz = self.vz_launch * (1.0 - 2.0 * local_phase)
            yaw_rate = -self.yaw_rate_compression * (1.0 - local_phase)  # Increasing to zero
        
        # Phase 0.8-1.0: Landing preparation
        else:
            local_phase = (phase - 0.8) / 0.2
            vx = self.vx_compression
            vy = -0.1 * self.vy_lateral * (1.0 - local_phase)  # Leftward then zero
            vz = self.vz_compression * local_phase  # Increasing downward velocity
            yaw_rate = self.yaw_rate_compression * local_phase  # Positive yaw preparation
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame based on phase and leg.
        
        Diagonal pairs:
        - Left diagonal: FL, RL (primary loading 0.0-0.2, 0.8-1.0)
        - Right diagonal: FR, RR (primary loading 0.4-0.6)
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Phase 0.0-0.2: Left compression
        if phase < 0.2:
            local_phase = phase / 0.2
            if is_left:
                # Primary loading legs: compress
                foot[2] += self.compression_depth * local_phase
                foot[0] -= 0.05 * local_phase  # Slight rearward shift
            else:
                # Secondary support: slight compression
                foot[2] += 0.5 * self.compression_depth * local_phase
        
        # Phase 0.2-0.4: Right-diagonal launch (aerial)
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # All feet airborne: extend and reposition
            swing_height = self.leg_lift_height * np.sin(np.pi * local_phase)
            foot[2] -= swing_height
            
            # Reposition forward and rightward for landing
            foot[0] += self.step_length_forward * 0.3 * local_phase
            if is_left:
                foot[1] += self.step_length_lateral * 0.2 * local_phase
            else:
                foot[1] += self.step_length_lateral * 0.4 * local_phase
        
        # Phase 0.4-0.6: Right compression
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            if not is_left:
                # Primary loading legs (FR, RR): compress
                foot[2] += self.compression_depth * local_phase
                foot[0] -= 0.05 * local_phase
            else:
                # Secondary support: slight compression
                foot[2] += 0.5 * self.compression_depth * local_phase
        
        # Phase 0.6-0.8: Left-diagonal launch (aerial)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # All feet airborne: extend and reposition
            swing_height = self.leg_lift_height * np.sin(np.pi * local_phase)
            foot[2] -= swing_height
            
            # Reposition forward and leftward for landing
            foot[0] += self.step_length_forward * 0.3 * local_phase
            if is_left:
                foot[1] -= self.step_length_lateral * 0.4 * local_phase
            else:
                foot[1] -= self.step_length_lateral * 0.2 * local_phase
        
        # Phase 0.8-1.0: Landing preparation
        else:
            local_phase = (phase - 0.8) / 0.2
            if is_left:
                # Preparing for primary loading: position and begin compression
                foot[2] += self.compression_depth * 0.3 * local_phase
                foot[0] += 0.05 * (1.0 - local_phase)
            else:
                # Secondary support positioning
                foot[2] += 0.3 * self.compression_depth * local_phase
        
        return foot