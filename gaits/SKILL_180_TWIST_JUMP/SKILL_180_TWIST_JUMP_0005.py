from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_180_TWIST_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    180-degree twist jump: explosive vertical launch with simultaneous yaw rotation.
    
    Phase structure:
      [0.0, 0.2]: Compression and yaw preload
      [0.2, 0.3]: Explosive launch
      [0.3, 0.7]: Aerial rotation with tucked legs
      [0.7, 0.8]: Landing preparation (leg extension)
      [0.8, 0.95]: Impact absorption
      [0.95, 1.0]: Stabilization
    
    All four legs are synchronized throughout the motion.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.compression_depth = 0.08  # meters, how much legs compress
        self.tuck_factor = 0.4  # fraction of leg length to tuck during aerial phase
        self.launch_vz = 1.2  # m/s, vertical launch velocity
        self.max_yaw_rate = 6.0  # rad/s, peak yaw rate to achieve 180 deg rotation
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocities based on phase.
        
        Phase-dependent behavior:
          - [0.0, 0.2]: Downward motion during compression, yaw rate builds
          - [0.2, 0.3]: Explosive upward velocity, peak yaw rate
          - [0.3, 0.7]: Ballistic flight with sustained yaw rate
          - [0.7, 0.8]: Descending, yaw rate decreases
          - [0.8, 0.95]: Landing impact absorption, yaw rate decays
          - [0.95, 1.0]: Stabilization, all velocities zero
        """
        
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Compression and preload
        if phase < 0.2:
            local_phase = phase / 0.2
            # Slight downward motion during compression
            vz = -0.3 * np.sin(np.pi * local_phase)
            # Yaw rate builds up smoothly
            yaw_rate = self.max_yaw_rate * 0.3 * local_phase
            
        # Explosive launch
        elif phase < 0.3:
            local_phase = (phase - 0.2) / 0.1
            # Strong upward velocity
            vz = self.launch_vz * (1.0 - 0.3 * local_phase)
            # Peak yaw rate
            yaw_rate = self.max_yaw_rate
            
        # Aerial rotation (ballistic arc)
        elif phase < 0.7:
            local_phase = (phase - 0.3) / 0.4
            # Ballistic vertical motion: starts positive, crosses zero at apex, becomes negative
            vz = self.launch_vz * (0.7 - 2.0 * local_phase)
            # Sustained high yaw rate to accumulate 180 degrees
            yaw_rate = self.max_yaw_rate * 0.95
            
        # Landing preparation
        elif phase < 0.8:
            local_phase = (phase - 0.7) / 0.1
            # Descending velocity
            vz = -self.launch_vz * (0.5 + 0.3 * local_phase)
            # Yaw rate begins decreasing
            yaw_rate = self.max_yaw_rate * 0.95 * (1.0 - local_phase)
            
        # Impact and absorption
        elif phase < 0.95:
            local_phase = (phase - 0.8) / 0.15
            # Rapid deceleration to zero
            vz = -self.launch_vz * 0.8 * (1.0 - local_phase) ** 2
            # Yaw rate decays rapidly
            yaw_rate = self.max_yaw_rate * 0.1 * (1.0 - local_phase) ** 2
            
        # Stabilization
        else:
            # All velocities zero
            vz = 0.0
            yaw_rate = 0.0
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
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
        Compute foot position in body frame for given leg and phase.
        
        All legs move synchronously:
          - Compression: feet move down and slightly inward
          - Launch: feet extend rapidly
          - Aerial tuck: feet retract toward body center
          - Landing prep: feet extend to landing position
          - Impact: feet compress again
          - Stabilization: return to nominal stance
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        base_x = foot[0]
        base_y = foot[1]
        base_z = foot[2]
        
        # Determine if front or rear leg for slight asymmetry handling
        is_front = leg_name.startswith('F')
        is_left = leg_name.endswith('L')
        
        # Compression and preload
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth compression curve
            compression = self.compression_depth * np.sin(np.pi * local_phase / 2.0)
            foot[2] = base_z + compression
            # Slight inward movement
            foot[0] = base_x * (1.0 - 0.05 * local_phase)
            foot[1] = base_y * (1.0 - 0.05 * local_phase)
            
        # Explosive launch
        elif phase < 0.3:
            local_phase = (phase - 0.2) / 0.1
            # Rapid extension from compressed state
            compression = self.compression_depth * (1.0 - local_phase)
            foot[2] = base_z + compression
            # Feet begin moving toward tuck position
            tuck_progress = local_phase * 0.3
            foot[0] = base_x * (0.95 - 0.2 * tuck_progress)
            foot[1] = base_y * (0.95 - 0.2 * tuck_progress)
            
        # Aerial rotation (tucked)
        elif phase < 0.7:
            local_phase = (phase - 0.3) / 0.4
            # Full tuck: legs drawn significantly inward and upward
            tuck_amount = self.tuck_factor * np.sin(np.pi * min(local_phase * 2.5, 1.0))
            foot[0] = base_x * (0.9 - tuck_amount)
            foot[1] = base_y * (0.9 - tuck_amount)
            foot[2] = base_z + tuck_amount * 0.15
            
        # Landing preparation
        elif phase < 0.8:
            local_phase = (phase - 0.7) / 0.1
            # Smooth extension to landing position
            extension_progress = np.sin(np.pi * local_phase / 2.0)
            tuck_amount = self.tuck_factor * (1.0 - extension_progress)
            foot[0] = base_x * (0.9 - tuck_amount + 0.1 * extension_progress)
            foot[1] = base_y * (0.9 - tuck_amount + 0.1 * extension_progress)
            foot[2] = base_z + tuck_amount * 0.15 * (1.0 - extension_progress)
            
        # Impact and absorption
        elif phase < 0.95:
            local_phase = (phase - 0.8) / 0.15
            # Compression during landing
            compression = self.compression_depth * np.sin(np.pi * local_phase / 2.0) * 1.2
            foot[2] = base_z + compression
            # Feet return to nominal lateral position
            foot[0] = base_x
            foot[1] = base_y
            
        # Stabilization
        else:
            local_phase = (phase - 0.95) / 0.05
            # Final settling to nominal stance
            compression = self.compression_depth * 1.2 * (1.0 - local_phase)
            foot[2] = base_z + compression
            foot[0] = base_x
            foot[1] = base_y
        
        return foot