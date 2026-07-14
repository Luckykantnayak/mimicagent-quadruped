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
        self.freq = 0.5  # Slower frequency for aerial skill (2 seconds per cycle)

        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Motion parameters
        self.crouch_depth = 0.12  # Vertical compression during crouch
        self.tuck_amount = 0.6  # Fraction to retract legs during aerial phase
        self.landing_extension = 0.08  # Additional leg extension for landing prep
        
        # Base velocity parameters
        self.takeoff_vx = 1.2  # Forward velocity during takeoff
        self.takeoff_vz = 2.5  # Upward velocity during takeoff
        self.aerial_vx = 1.0  # Sustained forward velocity during flight
        self.crouch_vz = -0.8  # Downward velocity during crouch
        
        # Angular velocity parameters
        self.yaw_rate = 6.0  # rad/s during twist phase to achieve ~180° rotation
        
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
            vz = self.crouch_vz
            
        elif phase < 0.4:
            # Explosive takeoff: forward and upward velocity
            local_phase = (phase - 0.2) / 0.2
            vx = self.takeoff_vx
            vz = self.takeoff_vz * (1.0 - local_phase)  # Taper upward velocity
            
        elif phase < 0.7:
            # Aerial twist: maintain forward velocity, apply yaw rotation
            vx = self.aerial_vx
            # Simulate ballistic arc: upward at start, downward at end
            local_phase = (phase - 0.4) / 0.3
            vz = self.takeoff_vz * 0.3 * (1.0 - 2.0 * local_phase)  # Parabolic apex
            yaw_rate = self.yaw_rate
            
        elif phase < 0.9:
            # Landing preparation: maintain forward velocity, descending
            local_phase = (phase - 0.7) / 0.2
            vx = self.aerial_vx * (1.0 - local_phase)  # Slow down
            vz = -1.5 * local_phase  # Accelerate downward
            
        else:
            # Impact absorption: rapidly decelerate to zero
            local_phase = (phase - 0.9) / 0.1
            vx = self.aerial_vx * 0.2 * (1.0 - local_phase)
            vz = -0.5 * (1.0 - local_phase)
        
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
            # Crouch preparation: compress legs vertically, feet move up toward body
            local_phase = phase / 0.2
            compression = self.crouch_depth * local_phase
            foot[2] += compression  # Move foot upward (leg shortens)
            
        elif phase < 0.4:
            # Explosive takeoff: extend legs downward, feet push away from body
            local_phase = (phase - 0.2) / 0.2
            # Start from compressed state, extend back toward base position
            compression = self.crouch_depth * (1.0 - local_phase)
            foot[2] += compression
            # Small backward push in body frame during extension
            foot[0] -= 0.03 * local_phase
            
        elif phase < 0.7:
            # Aerial twist: tuck legs close to body to reduce rotational inertia
            local_phase = (phase - 0.4) / 0.3
            # Smooth tuck using sinusoidal profile
            tuck_factor = self.tuck_amount * np.sin(np.pi * min(local_phase * 1.5, 1.0))
            foot[0] *= (1.0 - tuck_factor)  # Retract in x
            foot[1] *= (1.0 - tuck_factor)  # Retract in y
            foot[2] = base_pos[2] + 0.15 * tuck_factor  # Pull feet upward
            
        elif phase < 0.9:
            # Landing preparation: extend legs toward ground
            local_phase = (phase - 0.7) / 0.2
            # Smoothly extend from tucked to landing position
            tuck_factor = self.tuck_amount * (1.0 - local_phase)
            foot[0] = base_pos[0] * (1.0 - tuck_factor) + base_pos[0] * local_phase
            foot[1] = base_pos[1] * (1.0 - tuck_factor) + base_pos[1] * local_phase
            # Extend downward for landing
            foot[2] = base_pos[2] + 0.15 * tuck_factor - self.landing_extension * local_phase
            
        else:
            # Impact absorption: compress legs on landing
            local_phase = (phase - 0.9) / 0.1
            compression = self.crouch_depth * 0.8 * local_phase  # Absorb impact
            foot[2] += compression
        
        return foot