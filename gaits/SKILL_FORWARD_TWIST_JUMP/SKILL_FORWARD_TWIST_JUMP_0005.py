from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_FORWARD_TWIST_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Forward jump with 180-degree yaw rotation during flight.
    
    Phase structure:
      [0.0, 0.2]: Crouch preparation - all legs compress
      [0.2, 0.4]: Explosive takeoff - all legs extend, body launches
      [0.4, 0.7]: Aerial twist - airborne forward flight with 180° yaw rotation
      [0.7, 0.9]: Landing preparation - legs extend for touchdown
      [0.9, 1.0]: Touchdown absorption - all legs contact and compress
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Skill duration ~1.25 seconds
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.crouch_depth = 0.10  # Vertical compression during crouch (meters)
        self.tuck_height = 0.15   # Leg retraction during flight (meters)
        self.tuck_inward = 0.08   # Horizontal retraction toward body center
        
        # Takeoff parameters
        self.takeoff_vx = 1.5     # Forward velocity during launch (m/s)
        self.takeoff_vz = 2.0     # Upward velocity during launch (m/s)
        self.crouch_vz = -0.5     # Downward velocity during crouch (m/s)
        
        # Aerial rotation parameters
        self.yaw_rate_aerial = np.pi / 0.3  # rad/s to achieve π radians in 0.3 phase units
        
        # Landing parameters
        self.landing_vz = -1.5    # Downward velocity during descent (m/s)
        self.landing_compression = 0.08  # Leg compression on impact
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        if phase < 0.2:
            # Crouch preparation: body descends
            progress = phase / 0.2
            vz = self.crouch_vz * (1.0 - progress)  # Ramp down to zero
            
        elif phase < 0.4:
            # Explosive takeoff: forward and upward acceleration
            progress = (phase - 0.2) / 0.2
            vx = self.takeoff_vx * progress
            vz = self.takeoff_vz * progress
            
        elif phase < 0.7:
            # Aerial twist: sustained forward velocity, yaw rotation, ballistic vertical motion
            progress = (phase - 0.4) / 0.3
            vx = self.takeoff_vx
            # Ballistic arc: upward early, then downward
            vz = self.takeoff_vz * (1.0 - 2.0 * progress)
            yaw_rate = self.yaw_rate_aerial
            
        elif phase < 0.9:
            # Landing preparation: forward momentum continues, descending
            vx = self.takeoff_vx * 0.8  # Slight reduction
            vz = self.landing_vz
            yaw_rate = 0.0
            
        else:
            # Touchdown absorption: deceleration
            progress = (phase - 0.9) / 0.1
            vx = self.takeoff_vx * 0.8 * (1.0 - progress)
            vz = self.landing_vz * (1.0 - progress)
            yaw_rate = 0.0
        
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
        Compute foot position in body frame based on phase.
        All legs move synchronously.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if phase < 0.2:
            # Crouch: legs compress, foot appears higher in body frame
            progress = phase / 0.2
            foot[2] += self.crouch_depth * progress
            
        elif phase < 0.4:
            # Takeoff: legs extend rapidly downward
            progress = (phase - 0.2) / 0.2
            foot[2] += self.crouch_depth * (1.0 - progress)
            
        elif phase < 0.7:
            # Aerial: legs retract toward body center
            progress = (phase - 0.4) / 0.3
            # Smooth transition into tuck
            tuck_factor = np.sin(np.pi * min(progress, 0.5) / 0.5) if progress < 0.5 else 1.0
            
            # Retract inward (reduce x and y magnitude)
            foot[0] *= (1.0 - self.tuck_inward / abs(foot[0]) * tuck_factor) if foot[0] != 0 else 1.0
            foot[1] *= (1.0 - self.tuck_inward / abs(foot[1]) * tuck_factor) if foot[1] != 0 else 1.0
            # Lift up
            foot[2] += self.tuck_height * tuck_factor
            
        elif phase < 0.9:
            # Landing preparation: legs extend to nominal stance
            progress = (phase - 0.7) / 0.2
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            # Transition from tucked to nominal position
            base_foot = self.base_feet_pos_body[leg_name]
            tucked_x = base_foot[0] * (1.0 - self.tuck_inward / abs(base_foot[0])) if base_foot[0] != 0 else base_foot[0]
            tucked_y = base_foot[1] * (1.0 - self.tuck_inward / abs(base_foot[1])) if base_foot[1] != 0 else base_foot[1]
            tucked_z = base_foot[2] + self.tuck_height
            
            foot[0] = tucked_x + (base_foot[0] - tucked_x) * smooth_progress
            foot[1] = tucked_y + (base_foot[1] - tucked_y) * smooth_progress
            foot[2] = tucked_z + (base_foot[2] - tucked_z) * smooth_progress
            
        else:
            # Touchdown: legs compress to absorb impact
            progress = (phase - 0.9) / 0.1
            foot[2] += self.landing_compression * progress
        
        return foot