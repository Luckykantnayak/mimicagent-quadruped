from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_FORWARD_TWIST_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Forward jump with 180-degree yaw rotation.
    
    Phase breakdown:
      - [0.0, 0.2]: Crouch preparation (all legs compress)
      - [0.2, 0.4]: Explosive takeoff (all legs push off)
      - [0.4, 0.7]: Aerial twist (180-degree yaw rotation, legs tucked)
      - [0.7, 0.9]: Landing preparation (legs extend)
      - [0.9, 1.0]: Impact absorption (all legs contact ground)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.crouch_depth = 0.15  # Vertical compression during crouch
        self.tuck_horizontal = 0.12  # Horizontal distance feet move toward body center during tuck
        self.tuck_vertical = 0.10  # Vertical distance feet move upward during tuck
        self.landing_extension = 0.25  # Downward extension during landing preparation
        
        # Base motion parameters
        self.takeoff_vx = 1.5  # Forward velocity during takeoff
        self.takeoff_vz = 2.0  # Upward velocity during takeoff
        self.crouch_vz = -0.75  # Downward velocity during crouch
        self.landing_vz = -1.5  # Downward velocity during landing
        
        # Yaw rotation parameters
        self.yaw_rate_peak = 10.47
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocities according to phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.2: Crouch preparation
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth crouch velocity ramp
            vz = self.crouch_vz * np.sin(local_phase * np.pi / 2)
        
        # Phase 0.2-0.4: Explosive takeoff
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Smooth velocity profiles
            vx = self.takeoff_vx * np.sin(local_phase * np.pi / 2) * (1.0 - 0.3 * local_phase)
            vz = self.takeoff_vz * np.sin(local_phase * np.pi / 2) * (1.0 - 0.5 * local_phase)
        
        # Phase 0.4-0.7: Aerial twist
        elif phase < 0.7:
            local_phase = (phase - 0.4) / 0.3
            vx = self.takeoff_vx * 0.7
            # Smooth ballistic arc
            vz = self.takeoff_vz * 0.5 * np.cos(local_phase * np.pi)
            yaw_rate = self.yaw_rate_peak
        
        # Phase 0.7-0.9: Landing preparation
        elif phase < 0.9:
            local_phase = (phase - 0.7) / 0.2
            # Smooth deceleration
            vx = self.takeoff_vx * 0.7 * (1.0 - 0.5 * local_phase)
            # Smooth descent velocity ramp
            vz = self.landing_vz * np.sin(local_phase * np.pi / 2)
        
        # Phase 0.9-1.0: Impact absorption
        else:
            local_phase = (phase - 0.9) / 0.1
            # Smooth decay to zero
            vx = self.takeoff_vx * 0.35 * np.cos(local_phase * np.pi / 2)
            vz = self.landing_vz * np.cos(local_phase * np.pi / 2)
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        All legs move synchronously.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine if front or rear leg for sign adjustments
        is_front = leg_name.startswith('F')
        x_sign = 1.0 if is_front else -1.0
        
        # Phase 0.0-0.2: Crouch preparation
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth crouch with ease-in
            crouch_progress = np.sin(local_phase * np.pi / 2)
            foot[2] += self.crouch_depth * crouch_progress
            foot[0] -= x_sign * 0.02 * crouch_progress
        
        # Phase 0.2-0.4: Explosive takeoff
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Smooth extension from crouch
            crouch_amount = self.crouch_depth * np.cos(local_phase * np.pi / 2)
            foot[2] += crouch_amount
            foot[0] -= x_sign * 0.02 * np.cos(local_phase * np.pi / 2)
            # Smooth push-off
            foot[0] -= x_sign * 0.03 * np.sin(local_phase * np.pi / 2)
        
        # Phase 0.4-0.7: Aerial twist (legs tucked)
        elif phase < 0.7:
            local_phase = (phase - 0.4) / 0.3
            # Smooth tuck with ease-in-out
            tuck_progress = (1.0 - np.cos(local_phase * np.pi)) / 2.0
            foot[0] -= x_sign * self.tuck_horizontal * tuck_progress
            foot[2] += self.tuck_vertical * tuck_progress
            foot[1] *= (1.0 - 0.3 * tuck_progress)
        
        # Phase 0.7-0.9: Landing preparation (legs extend downward)
        elif phase < 0.9:
            local_phase = (phase - 0.7) / 0.2
            # Smooth untuck with ease-out
            tuck_progress = np.cos(local_phase * np.pi / 2)
            foot[0] -= x_sign * self.tuck_horizontal * tuck_progress
            foot[2] += self.tuck_vertical * tuck_progress
            foot[1] *= (1.0 - 0.3 * tuck_progress)
            
            # Critical: extend feet downward to compensate for descending base
            # Progressive extension reaching maximum at phase 0.9
            extension_progress = np.sin(local_phase * np.pi / 2)
            foot[2] -= self.landing_extension * extension_progress
        
        # Phase 0.9-1.0: Impact absorption
        else:
            local_phase = (phase - 0.9) / 0.1
            # Feet at maximum extension, then compress upward as legs absorb impact
            # Start at full extension, compress upward, modeling leg flexion under load
            compression_curve = np.sin(local_phase * np.pi)
            foot[2] -= self.landing_extension * (1.0 - 0.4 * compression_curve)
        
        return foot