from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Ricochet bounce diagonal gait with alternating diagonal bounces.
    
    Motion cycle:
    - Left compression with negative yaw (0.0-0.2)
    - Right diagonal launch aerial (0.2-0.4)
    - Right landing compression with positive yaw (0.4-0.6)
    - Left diagonal launch aerial (0.6-0.8)
    - Transition preparation (0.8-1.0)
    
    Base motion combines forward velocity with alternating diagonal components
    and yaw oscillations to create zigzag ricochet pattern.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.compression_depth = 0.12  # Vertical compression distance
        self.launch_height = 0.15  # Peak aerial height
        self.step_length_forward = 0.25  # Forward step per half-cycle
        self.step_width_lateral = 0.15  # Lateral displacement during diagonal
        self.yaw_amplitude = 0.52  # ~30 degrees in radians
        
        # Velocity parameters
        self.vx_cruise = 0.8  # Forward velocity baseline
        self.vx_launch = 2.0  # Peak forward velocity during launch
        self.vy_diagonal = 0.6  # Lateral velocity during diagonal motion
        self.vz_compression = -1.2  # Downward velocity during compression
        self.vz_launch = 2.5  # Upward velocity during launch
        self.yaw_rate_max = 3.0  # Peak yaw rate during rotation
        
        # Leg extension parameters
        self.leg_retract_swing = 0.08  # Leg retraction during swing
        self.leg_extend_aerial = 0.10  # Leg extension during aerial phase
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
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
            vx = self.vx_cruise * (1.0 - 0.5 * local_phase)
            vy = -0.2 * self.vy_diagonal * local_phase
            vz = self.vz_compression * np.sin(np.pi * local_phase)
            yaw_rate = -self.yaw_rate_max * np.sin(np.pi * local_phase)
        
        # Phase 0.2-0.4: Right diagonal launch aerial
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            vx = self.vx_cruise + (self.vx_launch - self.vx_cruise) * np.sin(np.pi * local_phase * 0.5)
            vy = self.vy_diagonal * np.sin(np.pi * local_phase)
            # Parabolic aerial trajectory
            if local_phase < 0.5:
                vz = self.vz_launch * (1.0 - 2.0 * local_phase)
            else:
                vz = -self.vz_launch * (2.0 * local_phase - 1.0)
            yaw_rate = -self.yaw_rate_max * 0.3 * (1.0 - local_phase)
        
        # Phase 0.4-0.6: Right landing compression
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            vx = self.vx_cruise * (1.0 - 0.5 * local_phase)
            vy = self.vy_diagonal * (1.0 - local_phase)
            vz = self.vz_compression * np.sin(np.pi * local_phase)
            yaw_rate = self.yaw_rate_max * np.sin(np.pi * local_phase)
        
        # Phase 0.6-0.8: Left diagonal launch aerial
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            vx = self.vx_cruise + (self.vx_launch - self.vx_cruise) * np.sin(np.pi * local_phase * 0.5)
            vy = -self.vy_diagonal * np.sin(np.pi * local_phase)
            # Parabolic aerial trajectory
            if local_phase < 0.5:
                vz = self.vz_launch * (1.0 - 2.0 * local_phase)
            else:
                vz = -self.vz_launch * (2.0 * local_phase - 1.0)
            yaw_rate = self.yaw_rate_max * 0.3 * (1.0 - local_phase)
        
        # Phase 0.8-1.0: Transition preparation
        else:
            local_phase = (phase - 0.8) / 0.2
            vx = self.vx_cruise * (0.5 + 0.5 * local_phase)
            vy = -self.vy_diagonal * (1.0 - local_phase) * 0.3
            vz = self.vz_compression * 0.5 * np.sin(np.pi * local_phase)
            yaw_rate = -self.yaw_rate_max * 0.4 * local_phase
        
        # Set velocity commands in world frame
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
        Compute foot position in body frame based on phase and leg role.
        
        Left legs (FL, RL): Primary load during left compression (0.0-0.2)
        Right legs (FR, RR): Primary load during right compression (0.4-0.6)
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Phase 0.0-0.2: Left compression
        if phase < 0.2:
            local_phase = phase / 0.2
            if is_left:
                # Primary load-bearing: compress in place
                foot[2] -= self.compression_depth * 0.5 * (1.0 - np.cos(np.pi * local_phase))
                foot[0] += 0.02 * local_phase  # Slight forward shift due to yaw
            else:
                # Secondary support: light compression
                foot[2] -= self.compression_depth * 0.3 * (1.0 - np.cos(np.pi * local_phase))
        
        # Phase 0.2-0.4: Right diagonal launch aerial
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # All legs extend during aerial phase
            foot[2] += self.leg_extend_aerial * np.sin(np.pi * local_phase)
            # Reposition for landing
            if is_left:
                foot[0] -= self.leg_retract_swing * local_phase
                foot[1] += 0.03 * local_phase  # Inward shift
            else:
                foot[0] -= self.leg_retract_swing * local_phase
                foot[1] -= 0.03 * local_phase  # Outward shift for landing prep
        
        # Phase 0.4-0.6: Right landing compression
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            if not is_left:
                # Primary load-bearing: compress significantly
                foot[2] -= self.compression_depth * 0.5 * (1.0 - np.cos(np.pi * local_phase))
                foot[0] -= 0.02 * local_phase  # Slight rearward shift due to yaw
            else:
                # Secondary support: light compression
                foot[2] -= self.compression_depth * 0.3 * (1.0 - np.cos(np.pi * local_phase))
        
        # Phase 0.6-0.8: Left diagonal launch aerial
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # All legs extend during aerial phase
            foot[2] += self.leg_extend_aerial * np.sin(np.pi * local_phase)
            # Reposition for landing
            if is_left:
                foot[0] += self.leg_retract_swing * 0.5 * local_phase
                foot[1] -= 0.03 * local_phase  # Outward shift for landing prep
            else:
                foot[0] += self.leg_retract_swing * 0.5 * local_phase
                foot[1] += 0.03 * local_phase  # Inward shift
        
        # Phase 0.8-1.0: Transition preparation
        else:
            local_phase = (phase - 0.8) / 0.2
            # All feet stabilize with light compression
            foot[2] -= self.compression_depth * 0.2 * (1.0 - np.cos(np.pi * local_phase))
            # Return to neutral positioning
            if is_left:
                foot[0] += 0.02 * local_phase
            else:
                foot[0] -= 0.01 * local_phase
        
        return foot