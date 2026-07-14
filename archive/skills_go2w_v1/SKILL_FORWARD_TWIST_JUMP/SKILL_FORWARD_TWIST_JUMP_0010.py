from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_FORWARD_TWIST_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Forward jump with 180-degree yaw rotation during flight.
    
    Phase breakdown:
      [0.0, 0.2]: compression - all legs compress, body lowers
      [0.2, 0.4]: takeoff - explosive extension, body launches forward and upward
      [0.4, 0.7]: aerial_twist - body rotates 180 degrees in yaw, legs tuck
      [0.7, 0.9]: landing_preparation - rotation completes, legs extend
      [0.9, 1.0]: landing - all feet contact, legs compress to absorb impact
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Skill duration ~2 seconds for realistic flight time
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Kinematic parameters
        self.compression_depth = 0.15  # How much legs compress (meters)
        self.tuck_factor = 0.5  # How much legs retract during aerial phase (0-1)
        
        # Velocity parameters for takeoff impulse
        self.takeoff_vx = 1.5  # Forward velocity (m/s)
        self.takeoff_vz = 2.5  # Upward velocity (m/s)
        
        # Yaw rotation parameters
        # Target: 180 degrees over phase range [0.4, 0.7] (duration 0.3)
        # Total yaw = pi radians
        self.yaw_rate_aerial = np.pi / (0.3 / self.freq)  # rad/s to achieve 180 deg
        
        # Gravity approximation for ballistic descent
        self.gravity = 9.81
        
    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on current phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 1: Compression [0.0, 0.2]
        if phase < 0.2:
            # Body descends as legs compress
            progress = phase / 0.2
            vz = -0.75 * np.sin(progress * np.pi)  # Smooth downward motion
            
        # Phase 2: Takeoff [0.2, 0.4]
        elif phase < 0.4:
            # Explosive upward and forward acceleration
            progress = (phase - 0.2) / 0.2
            # Smooth ramp-up to peak velocity
            vx = self.takeoff_vx * np.sin(progress * np.pi)
            vz = self.takeoff_vz * np.sin(progress * np.pi)
            
        # Phase 3: Aerial twist [0.4, 0.7]
        elif phase < 0.7:
            # Ballistic forward motion with gravity, plus yaw rotation
            progress = (phase - 0.4) / 0.3
            # Forward velocity maintained (no ground forces)
            vx = self.takeoff_vx * 0.8  # Slight decay for realism
            # Vertical velocity: starts positive, becomes negative (apex crossing)
            apex_phase = 0.5
            if progress < apex_phase:
                vz = self.takeoff_vz * 0.5 * (1 - progress / apex_phase)
            else:
                vz = -self.gravity * 0.3 * ((progress - apex_phase) / (1 - apex_phase))
            # Constant yaw rotation
            yaw_rate = self.yaw_rate_aerial
            
        # Phase 4: Landing preparation [0.7, 0.9]
        elif phase < 0.9:
            # Descending, rotation complete
            progress = (phase - 0.7) / 0.2
            vx = self.takeoff_vx * 0.5 * (1 - progress)  # Decelerate forward
            vz = -self.gravity * 0.4 * (1 + progress)  # Accelerate downward
            yaw_rate = 0.0  # Rotation complete
            
        # Phase 5: Landing [0.9, 1.0]
        else:
            # Rapid deceleration on impact
            progress = (phase - 0.9) / 0.1
            # Aggressive damping
            vx = 0.0
            vy = 0.0
            vz = 0.0
            yaw_rate = 0.0
        
        # Set velocities and integrate
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
        
        # Phase 1: Compression [0.0, 0.2]
        if phase < 0.2:
            progress = phase / 0.2
            # Legs compress: feet move upward relative to body
            # Smooth compression using sinusoidal profile
            compression = self.compression_depth * np.sin(progress * np.pi * 0.5)
            foot[2] += compression
            
        # Phase 2: Takeoff [0.2, 0.4]
        elif phase < 0.4:
            progress = (phase - 0.2) / 0.2
            # Legs extend explosively then lose contact
            # Foot moves down then begins to lift
            extension = self.compression_depth * (1 - np.sin((0.5 + progress * 0.5) * np.pi))
            foot[2] += extension
            
        # Phase 3: Aerial twist [0.4, 0.7]
        elif phase < 0.7:
            progress = (phase - 0.4) / 0.3
            # Legs tuck toward body center to reduce moment of inertia
            tuck = self.tuck_factor * np.sin(progress * np.pi)
            # Retract in horizontal plane toward body center
            foot[0] *= (1 - tuck * 0.6)
            foot[1] *= (1 - tuck * 0.6)
            # Lift feet upward
            foot[2] += self.compression_depth * 0.5 + tuck * 0.15
            
        # Phase 4: Landing preparation [0.7, 0.9]
        elif phase < 0.9:
            progress = (phase - 0.7) / 0.2
            # Legs extend back to nominal stance position
            # Smooth transition from tucked to extended
            tuck = self.tuck_factor * np.sin((1 - progress) * np.pi * 0.5)
            foot[0] = base_pos[0] * (1 - tuck * 0.6) + base_pos[0] * progress * tuck * 0.6
            foot[1] = base_pos[1] * (1 - tuck * 0.6) + base_pos[1] * progress * tuck * 0.6
            # Feet move downward toward landing position
            lift = (self.compression_depth * 0.5 + tuck * 0.15) * (1 - progress)
            foot[2] += lift
            
        # Phase 5: Landing [0.9, 1.0]
        else:
            progress = (phase - 0.9) / 0.1
            # Legs compress rapidly to absorb impact
            # Feet rise relative to body as legs yield
            impact_compression = self.compression_depth * 0.8 * np.sin(progress * np.pi * 0.5)
            foot[2] += impact_compression
        
        return foot