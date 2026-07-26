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
        
        # Base motion parameters
        self.takeoff_vx = 1.5  # Forward velocity during takeoff
        self.takeoff_vz = 2.0  # Upward velocity during takeoff
        self.crouch_vz = -0.75  # Downward velocity during crouch
        self.landing_vz = -1.5  # Downward velocity during landing
        
        # Yaw rotation parameters
        # Target: 180 degrees over phase 0.4 to 0.7 (duration 0.3 in phase units)
        # At freq=1.0, phase duration 0.3 corresponds to 0.3 seconds
        # yaw_rate * 0.3 = π  =>  yaw_rate ≈ 10.47 rad/s
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
            vz = self.crouch_vz
        
        # Phase 0.2-0.4: Explosive takeoff
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            vx = self.takeoff_vx * (1.0 - 0.3 * local_phase)  # Gradual decay
            vz = self.takeoff_vz * (1.0 - 0.5 * local_phase)  # Gradual decay
        
        # Phase 0.4-0.7: Aerial twist
        elif phase < 0.7:
            local_phase = (phase - 0.4) / 0.3
            vx = self.takeoff_vx * 0.7  # Maintain forward momentum
            # Ballistic arc: vz transitions from positive to negative
            vz = self.takeoff_vz * 0.5 * (1.0 - 2.0 * local_phase)
            yaw_rate = self.yaw_rate_peak
        
        # Phase 0.7-0.9: Landing preparation
        elif phase < 0.9:
            local_phase = (phase - 0.7) / 0.2
            vx = self.takeoff_vx * 0.7 * (1.0 - 0.5 * local_phase)
            vz = self.landing_vz * local_phase
        
        # Phase 0.9-1.0: Impact absorption
        else:
            local_phase = (phase - 0.9) / 0.1
            vx = self.takeoff_vx * 0.35 * (1.0 - local_phase)
            vz = self.landing_vz * (1.0 - local_phase)
        
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
            # Feet move upward (legs compress) and slightly inward
            foot[2] += self.crouch_depth * local_phase
            foot[0] -= x_sign * 0.02 * local_phase  # Slight inward shift
        
        # Phase 0.2-0.4: Explosive takeoff
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Feet extend back to base position and push downward
            crouch_amount = self.crouch_depth * (1.0 - local_phase)
            foot[2] += crouch_amount
            foot[0] -= x_sign * 0.02 * (1.0 - local_phase)
            # Push-off: feet move slightly backward relative to body
            foot[0] -= x_sign * 0.03 * local_phase
        
        # Phase 0.4-0.7: Aerial twist (legs tucked)
        elif phase < 0.7:
            local_phase = (phase - 0.4) / 0.3
            # Smooth tuck: feet move toward body center and upward
            tuck_progress = np.sin(local_phase * np.pi / 2)  # Smooth ease-in
            foot[0] -= x_sign * self.tuck_horizontal * tuck_progress
            foot[2] += self.tuck_vertical * tuck_progress
            foot[1] *= (1.0 - 0.3 * tuck_progress)  # Also tuck laterally
        
        # Phase 0.7-0.9: Landing preparation (legs extend)
        elif phase < 0.9:
            local_phase = (phase - 0.7) / 0.2
            # Extend from tucked to landing position
            tuck_progress = 1.0 - local_phase
            foot[0] -= x_sign * self.tuck_horizontal * tuck_progress
            foot[2] += self.tuck_vertical * tuck_progress
            foot[1] *= (1.0 - 0.3 * tuck_progress)
        
        # Phase 0.9-1.0: Impact absorption
        else:
            local_phase = (phase - 0.9) / 0.1
            # Legs compress slightly on landing
            compression = 0.08 * np.sin(local_phase * np.pi)
            foot[2] += compression
        
        return foot