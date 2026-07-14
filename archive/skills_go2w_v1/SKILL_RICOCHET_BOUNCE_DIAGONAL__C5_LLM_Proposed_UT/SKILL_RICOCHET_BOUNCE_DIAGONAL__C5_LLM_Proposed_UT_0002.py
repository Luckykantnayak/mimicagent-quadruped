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
        
        # Motion parameters - tuned to avoid ground penetration and maintain safe height
        self.compression_depth = 0.06  # Reduced vertical compression (m) - feet move UP in body frame
        self.nominal_base_height = 0.28  # Target base height above ground
        self.aerial_height = 0.10  # Peak aerial height above nominal
        self.step_length_forward = 0.20  # Forward displacement per cycle
        self.step_length_lateral = 0.12  # Lateral zigzag amplitude
        self.leg_lift_height = 0.08  # Foot clearance during aerial phase
        
        # Velocity parameters - rebalanced for safe height envelope
        self.vx_cruise = 0.7  # Forward velocity during aerial phase
        self.vx_compression = 0.35  # Forward velocity during compression
        self.vy_lateral = 0.5  # Lateral velocity magnitude
        self.vz_launch = 0.8  # Upward launch velocity (reduced for smoother motion)
        self.vz_compression = -0.35  # Downward compression velocity (reduced to prevent excessive descent)
        
        # Angular velocity parameters
        self.yaw_rate_compression = 2.2  # Yaw rate during compression (rad/s)
        self.max_yaw_angle = np.deg2rad(30)  # Maximum yaw deviation
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.nominal_base_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity state
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocities based on phase to create ricochet bounce pattern.
        Height regulation ensures base stays within safe envelope.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Height regulation: bias upward if too low, downward if too high
        height_error = self.root_pos[2] - self.nominal_base_height
        height_correction = -0.8 * height_error  # Proportional feedback
        
        # Phase 0.0-0.2: Left compression
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth compression using cosine profile
            compression_profile = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            vx = self.vx_compression
            vy = -0.08 * self.vy_lateral  # Slight leftward drift
            vz = self.vz_compression * compression_profile + height_correction
            yaw_rate = self.yaw_rate_compression * np.sin(np.pi * local_phase)  # Smooth yaw acceleration
        
        # Phase 0.2-0.4: Right-diagonal launch (aerial)
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            vx = self.vx_cruise
            vy = self.vy_lateral  # Rightward velocity
            # Launch profile: strong initial upward, then ballistic descent
            launch_profile = 1.0 - 1.5 * local_phase
            vz = self.vz_launch * launch_profile + height_correction
            # Yaw rate decays smoothly to zero
            yaw_rate = self.yaw_rate_compression * (1.0 - local_phase)**2
        
        # Phase 0.4-0.6: Right compression
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            compression_profile = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            vx = self.vx_compression
            vy = 0.08 * self.vy_lateral * (1.0 - local_phase)  # Slight rightward then zero
            vz = self.vz_compression * compression_profile + height_correction
            yaw_rate = -self.yaw_rate_compression * np.sin(np.pi * local_phase)  # Smooth negative yaw
        
        # Phase 0.6-0.8: Left-diagonal launch (aerial)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            vx = self.vx_cruise
            vy = -self.vy_lateral  # Leftward velocity
            launch_profile = 1.0 - 1.5 * local_phase
            vz = self.vz_launch * launch_profile + height_correction
            # Yaw rate returns smoothly to zero
            yaw_rate = -self.yaw_rate_compression * (1.0 - local_phase)**2
        
        # Phase 0.8-1.0: Landing preparation
        else:
            local_phase = (phase - 0.8) / 0.2
            vx = self.vx_compression
            vy = -0.08 * self.vy_lateral * (1.0 - local_phase)  # Leftward then zero
            # Gentle descent for landing
            vz = -0.2 * local_phase + height_correction
            # Prepare positive yaw for next cycle
            yaw_rate = self.yaw_rate_compression * local_phase * 0.5
        
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
        
        CRITICAL: During compression, feet move UPWARD (negative z) in body frame
        to represent leg retraction. This prevents ground penetration.
        
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
            # Smooth compression profile
            compression_factor = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            
            if is_left:
                # Primary loading legs: compress (feet move UP toward base)
                foot[2] -= self.compression_depth * compression_factor
                foot[0] -= 0.03 * compression_factor  # Slight rearward shift
            else:
                # Secondary support: lighter compression
                foot[2] -= 0.4 * self.compression_depth * compression_factor
        
        # Phase 0.2-0.4: Right-diagonal launch (aerial)
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # All feet airborne: lift and reposition
            swing_height = self.leg_lift_height * np.sin(np.pi * local_phase)
            foot[2] -= swing_height  # Lift feet upward
            
            # Reposition forward and rightward for landing
            foot[0] += self.step_length_forward * 0.25 * local_phase
            if is_left:
                foot[1] += self.step_length_lateral * 0.15 * local_phase
            else:
                foot[1] += self.step_length_lateral * 0.35 * local_phase
        
        # Phase 0.4-0.6: Right compression
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            compression_factor = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            
            if not is_left:
                # Primary loading legs (FR, RR): compress (feet move UP)
                foot[2] -= self.compression_depth * compression_factor
                foot[0] -= 0.03 * compression_factor
            else:
                # Secondary support: lighter compression
                foot[2] -= 0.4 * self.compression_depth * compression_factor
        
        # Phase 0.6-0.8: Left-diagonal launch (aerial)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # All feet airborne: lift and reposition
            swing_height = self.leg_lift_height * np.sin(np.pi * local_phase)
            foot[2] -= swing_height
            
            # Reposition forward and leftward for landing
            foot[0] += self.step_length_forward * 0.25 * local_phase
            if is_left:
                foot[1] -= self.step_length_lateral * 0.35 * local_phase
            else:
                foot[1] -= self.step_length_lateral * 0.15 * local_phase
        
        # Phase 0.8-1.0: Landing preparation
        else:
            local_phase = (phase - 0.8) / 0.2
            # Smooth transition to neutral stance
            transition_factor = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            
            if is_left:
                # Preparing for primary loading: slight pre-compression
                foot[2] -= 0.3 * self.compression_depth * transition_factor
                foot[0] += 0.02 * (1.0 - local_phase)
            else:
                # Secondary support positioning
                foot[2] -= 0.2 * self.compression_depth * transition_factor
        
        return foot