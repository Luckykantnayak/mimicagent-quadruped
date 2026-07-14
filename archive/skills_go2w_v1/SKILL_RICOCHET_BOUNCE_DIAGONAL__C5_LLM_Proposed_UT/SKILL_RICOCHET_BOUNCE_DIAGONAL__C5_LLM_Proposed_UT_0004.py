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
        
        # Calculate safe nominal height based on initial foot positions
        # Increased margin from 0.08m to 0.12m for better ground clearance
        max_foot_depth = max([-pos[2] for pos in initial_foot_positions_body.values()])
        self.nominal_base_height = max(0.35, max_foot_depth + 0.12)  # Minimum 0.35m with 12cm margin
        
        # Motion parameters - balanced for safety and zigzag pattern
        self.compression_depth = 0.04  # Reduced to prevent joint limits while maintaining compression effect
        self.aerial_height = 0.08  # Peak aerial height above nominal
        self.step_length_forward = 0.18  # Forward displacement per cycle
        self.step_length_lateral = 0.12  # Lateral zigzag amplitude (restored from iteration 2)
        self.leg_lift_height = 0.06  # Foot clearance during aerial phase
        
        # Velocity parameters - restored lateral strength from iteration 2
        self.vx_cruise = 0.65  # Forward velocity during aerial phase
        self.vx_compression = 0.40  # Forward velocity during compression
        self.vy_lateral = 0.52  # Lateral velocity magnitude (restored and slightly increased)
        self.vz_launch = 0.65  # Upward launch velocity
        self.vz_compression = -0.30  # Downward compression velocity
        
        # Angular velocity parameters - restored from iteration 2
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
        Lateral velocities strengthened to create bidirectional zigzag.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Height regulation: gentle bias to maintain nominal height
        height_error = self.root_pos[2] - self.nominal_base_height
        height_correction = -0.5 * height_error  # Gentle proportional feedback
        
        # Phase 0.0-0.2: Left compression
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth compression using sine profile
            compression_profile = np.sin(0.5 * np.pi * local_phase)
            vx = self.vx_compression
            # Increased leftward drift to prepare for rightward launch
            vy = -0.18 * self.vy_lateral  # Strengthened from 0.06 to 0.18
            vz = self.vz_compression * compression_profile + height_correction
            # Smooth yaw acceleration
            yaw_rate = self.yaw_rate_compression * np.sin(np.pi * local_phase)
        
        # Phase 0.2-0.4: Right-diagonal launch (aerial)
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            vx = self.vx_cruise
            vy = self.vy_lateral  # Strong rightward velocity
            # Launch profile: strong initial upward, then gradual descent
            launch_profile = 1.0 - 1.2 * local_phase
            vz = self.vz_launch * launch_profile + height_correction
            # Yaw rate decays smoothly to zero
            yaw_rate = self.yaw_rate_compression * (1.0 - local_phase) * (1.0 - local_phase)
        
        # Phase 0.4-0.6: Right compression
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            compression_profile = np.sin(0.5 * np.pi * local_phase)
            vx = self.vx_compression
            # Increased rightward drift to prepare for leftward launch
            vy = 0.18 * self.vy_lateral * (1.0 - 0.5 * local_phase)  # Strengthened and sustained longer
            vz = self.vz_compression * compression_profile + height_correction
            # Smooth negative yaw
            yaw_rate = -self.yaw_rate_compression * np.sin(np.pi * local_phase)
        
        # Phase 0.6-0.8: Left-diagonal launch (aerial)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            vx = self.vx_cruise
            vy = -self.vy_lateral  # Strong leftward velocity
            launch_profile = 1.0 - 1.2 * local_phase
            vz = self.vz_launch * launch_profile + height_correction
            # Yaw rate returns smoothly to zero
            yaw_rate = -self.yaw_rate_compression * (1.0 - local_phase) * (1.0 - local_phase)
        
        # Phase 0.8-1.0: Landing preparation
        else:
            local_phase = (phase - 0.8) / 0.2
            vx = self.vx_compression
            # Increased leftward drift to prepare for next cycle
            vy = -0.18 * self.vy_lateral * (1.0 - 0.5 * local_phase)  # Strengthened and sustained
            # Gentle descent for landing
            vz = -0.15 * local_phase + height_correction
            # Prepare positive yaw for next cycle
            yaw_rate = self.yaw_rate_compression * local_phase * 0.4
        
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
        to represent leg retraction. Universal upward offset applied for ground clearance.
        
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
            compression_factor = np.sin(0.5 * np.pi * local_phase)
            
            if is_left:
                # Primary loading legs: compress (feet move UP toward base)
                foot[2] -= self.compression_depth * compression_factor
                foot[0] -= 0.02 * compression_factor  # Slight rearward shift
            else:
                # Secondary support: lighter compression
                foot[2] -= 0.3 * self.compression_depth * compression_factor
        
        # Phase 0.2-0.4: Right-diagonal launch (aerial)
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # All feet airborne: lift and reposition
            swing_height = self.leg_lift_height * np.sin(np.pi * local_phase)
            foot[2] -= swing_height  # Lift feet upward
            
            # Reposition forward and rightward for landing
            forward_shift = self.step_length_forward * 0.22 * (local_phase * local_phase)
            foot[0] += forward_shift
            
            if is_left:
                lateral_shift = self.step_length_lateral * 0.15 * (local_phase * local_phase)
                foot[1] += lateral_shift
            else:
                lateral_shift = self.step_length_lateral * 0.35 * (local_phase * local_phase)
                foot[1] += lateral_shift
        
        # Phase 0.4-0.6: Right compression
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            compression_factor = np.sin(0.5 * np.pi * local_phase)
            
            if not is_left:
                # Primary loading legs (FR, RR): compress (feet move UP)
                foot[2] -= self.compression_depth * compression_factor
                foot[0] -= 0.02 * compression_factor
            else:
                # Secondary support: lighter compression
                foot[2] -= 0.3 * self.compression_depth * compression_factor
        
        # Phase 0.6-0.8: Left-diagonal launch (aerial)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # All feet airborne: lift and reposition
            swing_height = self.leg_lift_height * np.sin(np.pi * local_phase)
            foot[2] -= swing_height
            
            # Reposition forward and leftward for landing
            forward_shift = self.step_length_forward * 0.22 * (local_phase * local_phase)
            foot[0] += forward_shift
            
            if is_left:
                lateral_shift = self.step_length_lateral * 0.35 * (local_phase * local_phase)
                foot[1] -= lateral_shift
            else:
                lateral_shift = self.step_length_lateral * 0.15 * (local_phase * local_phase)
                foot[1] -= lateral_shift
        
        # Phase 0.8-1.0: Landing preparation
        else:
            local_phase = (phase - 0.8) / 0.2
            # Smooth transition to neutral stance
            transition_factor = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            
            if is_left:
                # Preparing for primary loading: slight pre-compression
                foot[2] -= 0.2 * self.compression_depth * transition_factor
                foot[0] += 0.01 * (1.0 - local_phase)
            else:
                # Secondary support positioning
                foot[2] -= 0.12 * self.compression_depth * transition_factor
        
        # Universal upward offset for additional ground clearance safety margin
        foot[2] -= 0.025  # 2.5cm additional lift for all feet
        
        return foot