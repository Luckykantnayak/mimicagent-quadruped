from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_FORWARD_TWIST_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Forward jump with 180-degree yaw rotation during aerial phase.
    
    Motion phases:
    - [0.0, 0.2]: Crouch preparation (all legs compress)
    - [0.2, 0.4]: Explosive takeoff (all legs extend, launch upward/forward)
    - [0.4, 0.7]: Aerial twist (body rotates 180° in yaw, legs tucked)
    - [0.7, 0.9]: Landing preparation (legs extend toward ground)
    - [0.9, 1.0]: Impact absorption (all legs compress on landing)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5

        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Motion parameters - tuned for kinematic system
        self.crouch_depth = 0.08
        self.tuck_amount = 0.45
        self.landing_extension = 0.18
        
        # Base velocity parameters - reduced for kinematic integration
        self.takeoff_vx = 1.1
        self.takeoff_vz = 1.1
        self.aerial_vx = 0.9
        self.crouch_vz = -0.6
        
        # Angular velocity parameters
        self.yaw_rate = 6.0
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base velocities based on jump phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        if phase < 0.2:
            # Crouch preparation: move downward
            local_phase = phase / 0.2
            vz = self.crouch_vz * smooth_step(local_phase)
            
        elif phase < 0.4:
            # Explosive takeoff: forward and upward velocity
            local_phase = (phase - 0.2) / 0.2
            vx = self.takeoff_vx * smooth_step(local_phase)
            vz = self.takeoff_vz * (1.0 - smooth_step(local_phase))
            
        elif phase < 0.7:
            # Aerial twist: maintain forward velocity, ballistic arc
            local_phase = (phase - 0.4) / 0.3
            vx = self.aerial_vx
            # Transition from slight upward to downward (ballistic descent)
            vz = 0.3 - 1.8 * local_phase
            yaw_rate = self.yaw_rate
            
        elif phase < 0.9:
            # Landing preparation: decelerate forward, accelerate downward
            local_phase = (phase - 0.7) / 0.2
            vx = self.aerial_vx * (1.0 - 0.7 * smooth_step(local_phase))
            vz = -1.5 - 1.0 * smooth_step(local_phase)
            
        else:
            # Impact absorption: rapidly decelerate to zero
            local_phase = (phase - 0.9) / 0.1
            decay = 1.0 - smooth_step(local_phase)
            vx = 0.3 * decay
            vz = -0.8 * decay
        
        # Set velocities in world frame
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
        Compute foot position in body frame for synchronized jumping motion.
        All legs move identically (symmetric behavior).
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        if phase < 0.2:
            # Crouch preparation: keep feet stationary in body frame while base lowers
            # Slight inward movement to represent compression
            local_phase = phase / 0.2
            smooth_phase = smooth_step(local_phase)
            foot[0] *= (1.0 - 0.1 * smooth_phase)
            foot[1] *= (1.0 - 0.1 * smooth_phase)
            
        elif phase < 0.4:
            # Explosive takeoff: feet push backward and down in body frame
            local_phase = (phase - 0.2) / 0.2
            smooth_phase = smooth_step(local_phase)
            # Return from compressed to base, then extend slightly backward
            foot[0] = base_pos[0] * (0.9 + 0.1 * (1.0 - smooth_phase)) - 0.04 * smooth_phase
            foot[1] = base_pos[1] * (0.9 + 0.1 * (1.0 - smooth_phase))
            foot[2] = base_pos[2] - 0.05 * smooth_phase
            
        elif phase < 0.7:
            # Aerial twist: tuck legs close to body
            local_phase = (phase - 0.4) / 0.3
            smooth_phase = smooth_step(local_phase)
            tuck_factor = self.tuck_amount * np.sin(np.pi * smooth_phase)
            foot[0] = base_pos[0] * (1.0 - tuck_factor)
            foot[1] = base_pos[1] * (1.0 - tuck_factor)
            foot[2] = base_pos[2] + 0.12 * tuck_factor
            
        elif phase < 0.9:
            # Landing preparation: extend legs aggressively toward ground
            local_phase = (phase - 0.7) / 0.2
            smooth_phase = smooth_step(local_phase)
            # Transition from tucked to extended landing position
            tuck_factor = self.tuck_amount * np.sin(np.pi * (1.0 - smooth_phase))
            foot[0] = base_pos[0] * (1.0 - tuck_factor)
            foot[1] = base_pos[1] * (1.0 - tuck_factor)
            # Aggressive downward extension for landing
            foot[2] = base_pos[2] + 0.12 * tuck_factor - self.landing_extension * smooth_phase
            
        else:
            # Impact absorption: feet remain at landing position, slight compression
            local_phase = (phase - 0.9) / 0.1
            smooth_phase = smooth_step(local_phase)
            compression = self.crouch_depth * 0.5 * smooth_phase
            foot[0] = base_pos[0] * (1.0 + 0.05 * smooth_phase)
            foot[1] = base_pos[1] * (1.0 + 0.05 * smooth_phase)
            foot[2] = base_pos[2] - self.landing_extension + compression
        
        return foot