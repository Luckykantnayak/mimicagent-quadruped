from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_FRONT_FLIP_MotionGenerator(BaseMotionGenerator):
    """
    Front flip motion: The robot performs a complete forward somersault.
    
    Phase breakdown:
      [0.0, 0.15]: Preparation crouch - all legs compress, base lowers
      [0.15, 0.30]: Launch and takeoff - rear legs extend explosively, front legs retract, base lifts off
      [0.30, 0.70]: Aerial rotation - all feet off ground, body rotates forward ~360 degrees
      [0.70, 0.80]: Landing preparation - legs extend anticipatorily, still airborne
      [0.80, 1.0]: Impact and stabilization - all legs contact ground, absorb impact, return to stance
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for single flip cycle
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.crouch_depth = 0.08  # How much legs retract during crouch
        self.launch_height = 0.20  # Maximum leg extension during launch
        self.tuck_amount = 0.12  # How much legs tuck during aerial phase
        self.landing_extension = 0.05  # Pre-landing leg extension
        
        # Base motion parameters
        self.launch_vz = 2.5  # Upward velocity during launch
        self.launch_pitch_rate = 8.0  # Forward pitch rate during launch (rad/s)
        self.aerial_pitch_rate = 6.0  # Sustained pitch rate during aerial phase
        self.gravity_vz = -1.5  # Effective downward velocity during aerial descent
        
    def update_base_motion(self, phase, dt):
        """
        Update base motion through all phases of the front flip.
        """
        
        # Phase 1: Preparation crouch [0.0, 0.15]
        if phase < 0.15:
            progress = phase / 0.15
            # Slight downward motion as legs compress
            vz = -0.5 * np.sin(np.pi * progress) if progress < 0.5 else 0.0
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, 0.0, 0.0])
        
        # Phase 2: Launch and takeoff [0.15, 0.30]
        elif phase < 0.30:
            progress = (phase - 0.15) / 0.15
            # Explosive upward velocity and forward pitch rate
            vz = self.launch_vz * (1.0 - progress * 0.3)  # Decay slightly as launch completes
            pitch_rate = self.launch_pitch_rate * np.sin(np.pi * progress * 0.5)  # Smooth ramp up
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 3: Aerial rotation [0.30, 0.70]
        elif phase < 0.70:
            progress = (phase - 0.30) / 0.40
            # Parabolic trajectory: upward velocity decays, then downward
            vz = self.launch_vz * 0.7 * (1.0 - progress) + self.gravity_vz * progress
            # Sustained pitch rate to complete 360 degree rotation
            pitch_rate = self.aerial_pitch_rate
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 4: Landing preparation [0.70, 0.80]
        elif phase < 0.80:
            progress = (phase - 0.70) / 0.10
            # Downward velocity, pitch rate begins to decelerate
            vz = self.gravity_vz
            pitch_rate = self.aerial_pitch_rate * (1.0 - progress)
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 5: Impact and stabilization [0.80, 1.0]
        else:
            progress = (phase - 0.80) / 0.20
            # Rapid deceleration to zero
            decay = np.exp(-5.0 * progress)
            vz = self.gravity_vz * 0.3 * decay
            pitch_rate = 0.0
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
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
        Compute foot position in body frame for each leg through all phases.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        is_rear = leg_name.startswith('R')
        
        # Phase 1: Preparation crouch [0.0, 0.15]
        if phase < 0.15:
            progress = phase / 0.15
            # All legs retract upward (compress)
            retraction = self.crouch_depth * np.sin(np.pi * progress * 0.5)
            foot[2] += retraction
        
        # Phase 2: Launch and takeoff [0.15, 0.30]
        elif phase < 0.30:
            progress = (phase - 0.15) / 0.15
            
            if is_rear:
                # Rear legs: explosive extension downward then retract
                if progress < 0.5:
                    # Extension phase
                    extension = -self.launch_height * np.sin(np.pi * progress)
                    foot[2] += self.crouch_depth + extension
                else:
                    # Retraction phase
                    local_progress = (progress - 0.5) / 0.5
                    retract = self.crouch_depth * (1.0 - local_progress) - self.launch_height * (1.0 - np.sin(np.pi * (progress)))
                    foot[2] += retract
            else:
                # Front legs: retract sharply upward and rearward
                retract_z = self.crouch_depth + self.tuck_amount * progress
                retract_x = -0.05 * progress  # Pull rearward slightly
                foot[2] += retract_z
                foot[0] += retract_x
        
        # Phase 3: Aerial rotation [0.30, 0.70]
        elif phase < 0.70:
            progress = (phase - 0.30) / 0.40
            # All legs tucked close to body
            if is_front:
                foot[2] += self.crouch_depth + self.tuck_amount
                foot[0] += -0.05
            else:
                foot[2] += self.tuck_amount * 0.8
                foot[0] += 0.03  # Rear legs slightly forward during tuck
        
        # Phase 4: Landing preparation [0.70, 0.80]
        elif phase < 0.80:
            progress = (phase - 0.70) / 0.10
            # Legs extend toward landing positions
            if is_front:
                tuck_z = self.crouch_depth + self.tuck_amount
                tuck_x = -0.05
                extension_z = -self.landing_extension
                foot[2] += tuck_z + (extension_z - tuck_z) * progress
                foot[0] += tuck_x * (1.0 - progress)
            else:
                tuck_z = self.tuck_amount * 0.8
                tuck_x = 0.03
                extension_z = -self.landing_extension
                foot[2] += tuck_z + (extension_z - tuck_z) * progress
                foot[0] += tuck_x * (1.0 - progress)
        
        # Phase 5: Impact and stabilization [0.80, 1.0]
        else:
            progress = (phase - 0.80) / 0.20
            # Legs compress to absorb impact, then return to nominal stance
            if progress < 0.5:
                # Compression phase
                local_progress = progress / 0.5
                compression = self.crouch_depth * 0.6 * np.sin(np.pi * local_progress * 0.5)
                foot[2] += -self.landing_extension + compression
            else:
                # Recovery phase
                local_progress = (progress - 0.5) / 0.5
                compression = self.crouch_depth * 0.6 * (1.0 - local_progress)
                foot[2] += -self.landing_extension + compression
        
        return foot