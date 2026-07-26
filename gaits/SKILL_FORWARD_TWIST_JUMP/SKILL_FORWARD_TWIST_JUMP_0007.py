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
        
        # Motion parameters - tuned for envelope compliance
        self.crouch_depth = 0.05  # Conservative to prevent excessive base lowering
        self.tuck_height = 0.12   # Moderate retraction for aerial clearance
        self.tuck_inward = 0.06   # Conservative tucking toward body center
        
        # Takeoff parameters - balanced for altitude and envelope
        self.takeoff_vx = 1.3     # Forward velocity during launch
        self.takeoff_vz = 2.4     # Tuned for peak altitude within envelope (reduced from 2.8)
        
        # Aerial rotation parameters
        self.yaw_rate_aerial = np.pi / 0.3  # rad/s to achieve π radians in 0.3 phase units
        
        # Landing parameters
        self.landing_vz_peak = -1.2  # Controlled descent velocity
        self.landing_compression = 0.04  # Minimal compression to prevent excessive lowering
        
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
            # Crouch preparation: minimal downward velocity
            progress = phase / 0.2
            smooth_progress = smooth_step(progress)
            vz = -0.25 * (1.0 - smooth_progress)
            
        elif phase < 0.4:
            # Explosive takeoff: forward and upward acceleration
            progress = (phase - 0.2) / 0.2
            smooth_progress = smooth_step(progress)
            vx = self.takeoff_vx * smooth_progress
            vz = self.takeoff_vz * smooth_progress
            
        elif phase < 0.7:
            # Aerial twist: controlled ballistic trajectory with earlier descent
            progress = (phase - 0.4) / 0.3
            vx = self.takeoff_vx
            
            # Linear ballistic decay: vz crosses zero at progress ~0.4, reaches negative by end
            # vz = takeoff_vz * (1.0 - 2.5 * progress)
            # At progress=0: vz = 2.4 m/s
            # At progress=0.4: vz = 0 m/s
            # At progress=1.0: vz = -1.5 m/s
            vz = self.takeoff_vz * (1.0 - 2.5 * progress)
            
            yaw_rate = self.yaw_rate_aerial
            
        elif phase < 0.9:
            # Landing preparation: sustained forward, controlled descent
            progress = (phase - 0.7) / 0.2
            smooth_progress = smooth_step(progress)
            vx = self.takeoff_vx * 0.7
            # Smooth transition to landing descent velocity
            vz = -self.landing_vz_peak * smooth_progress
            yaw_rate = 0.0
            
        else:
            # Touchdown absorption: decelerate to rest
            progress = (phase - 0.9) / 0.1
            smooth_decay = 1.0 - smooth_step(progress)
            vx = self.takeoff_vx * 0.7 * smooth_decay
            vz = -self.landing_vz_peak * smooth_decay
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
        base_foot = self.base_feet_pos_body[leg_name]
        foot = base_foot.copy()
        
        if phase < 0.2:
            # Crouch: legs compress smoothly
            progress = phase / 0.2
            smooth_progress = smooth_step(progress)
            foot[2] += self.crouch_depth * smooth_progress
            
        elif phase < 0.4:
            # Takeoff: legs extend rapidly downward
            progress = (phase - 0.2) / 0.2
            smooth_progress = smooth_step(progress)
            foot[2] += self.crouch_depth * (1.0 - smooth_progress)
            
        elif phase < 0.7:
            # Aerial: legs retract toward body center using explicit offsets
            progress = (phase - 0.4) / 0.3
            # Smooth entry into tuck
            tuck_factor = smooth_step(min(progress * 2.0, 1.0))
            
            # Explicit tucking: move toward body center
            sign_x = np.sign(base_foot[0]) if base_foot[0] != 0 else 0
            sign_y = np.sign(base_foot[1]) if base_foot[1] != 0 else 0
            
            foot[0] = base_foot[0] - sign_x * self.tuck_inward * tuck_factor
            foot[1] = base_foot[1] - sign_y * self.tuck_inward * tuck_factor
            foot[2] = base_foot[2] + self.tuck_height * tuck_factor
            
        elif phase < 0.9:
            # Landing preparation: legs extend from tucked to nominal stance
            progress = (phase - 0.7) / 0.2
            smooth_progress = smooth_step(progress)
            
            # Define tucked position explicitly
            sign_x = np.sign(base_foot[0]) if base_foot[0] != 0 else 0
            sign_y = np.sign(base_foot[1]) if base_foot[1] != 0 else 0
            tucked_x = base_foot[0] - sign_x * self.tuck_inward
            tucked_y = base_foot[1] - sign_y * self.tuck_inward
            tucked_z = base_foot[2] + self.tuck_height
            
            # Interpolate from tucked to nominal
            foot[0] = tucked_x + (base_foot[0] - tucked_x) * smooth_progress
            foot[1] = tucked_y + (base_foot[1] - tucked_y) * smooth_progress
            foot[2] = tucked_z + (base_foot[2] - tucked_z) * smooth_progress
            
        else:
            # Touchdown: legs compress to absorb impact
            progress = (phase - 0.9) / 0.1
            smooth_progress = smooth_step(progress)
            foot[2] += self.landing_compression * smooth_progress
        
        return foot


def smooth_step(t):
    """Smooth interpolation function (smoothstep)."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)