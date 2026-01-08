from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_180_TWIST_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    180-degree twist jump: explosive vertical launch with simultaneous yaw rotation.
    
    Phase structure:
      [0.0, 0.2]: Compression and yaw preload
      [0.2, 0.3]: Explosive launch
      [0.3, 0.65]: Aerial rotation with tucked legs
      [0.65, 0.85]: Landing preparation (leg extension)
      [0.85, 0.95]: Impact absorption
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
        self.tuck_factor = 0.30  # reduced from 0.4 for faster extension
        self.launch_vz = 1.0  # m/s, reduced for lower apex and safer landing
        self.max_yaw_rate = 6.0  # rad/s, peak yaw rate to achieve 180 deg rotation
        self.g_sim = 2.5  # simulated gravity for parabolic trajectory
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocities based on phase with physically consistent ballistic trajectory.
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
            # Smooth downward motion during compression using cosine for continuity
            vz = -0.25 * (1.0 - np.cos(np.pi * local_phase)) * 0.5
            # Yaw rate builds up smoothly
            yaw_rate = self.max_yaw_rate * 0.3 * (local_phase ** 2)
            
        # Explosive launch
        elif phase < 0.3:
            local_phase = (phase - 0.2) / 0.1
            # Smooth transition to peak upward velocity
            vz = self.launch_vz * np.sin(np.pi * local_phase / 2.0)
            # Yaw rate increases to peak
            yaw_rate = self.max_yaw_rate * (0.3 + 0.7 * local_phase)
            
        # Aerial rotation (ballistic arc with parabolic velocity decay)
        elif phase < 0.65:
            local_phase = (phase - 0.3) / 0.35
            # Parabolic ballistic trajectory: v(t) = v0 - g*t
            # Normalized so apex occurs mid-aerial phase
            time_in_air = local_phase * 0.35 / self.freq
            vz = self.launch_vz - self.g_sim * time_in_air * 3.0
            # Sustained high yaw rate to accumulate 180 degrees
            yaw_rate = self.max_yaw_rate
            
        # Landing preparation (extended phase for proper foot extension)
        elif phase < 0.85:
            local_phase = (phase - 0.65) / 0.2
            # Continue ballistic descent but begin deceleration for landing
            # Smooth transition from ballistic to controlled descent
            ballistic_vz = -self.launch_vz * 0.65
            landing_vz = -0.4
            blend = np.sin(np.pi * local_phase / 2.0)
            vz = ballistic_vz * (1.0 - blend) + landing_vz * blend
            # Yaw rate begins decreasing smoothly
            yaw_rate = self.max_yaw_rate * (1.0 - 0.5 * local_phase)
            
        # Impact and absorption
        elif phase < 0.95:
            local_phase = (phase - 0.85) / 0.1
            # Rapid smooth deceleration to zero using cubic ease-out
            vz = -0.4 * (1.0 - local_phase) ** 3
            # Yaw rate decays rapidly
            yaw_rate = self.max_yaw_rate * 0.5 * (1.0 - local_phase) ** 3
            
        # Stabilization
        else:
            local_phase = (phase - 0.95) / 0.05
            # Final smooth approach to zero
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
        Compute foot position in body frame with extended landing preparation phase.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        base_x = foot[0]
        base_y = foot[1]
        base_z = foot[2]
        
        # Compression and preload
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth compression curve with continuity
            compression = self.compression_depth * (local_phase ** 2)
            foot[2] = base_z + compression
            # Slight inward movement using smooth curve
            inward_factor = 0.05 * (local_phase ** 2)
            foot[0] = base_x * (1.0 - inward_factor)
            foot[1] = base_y * (1.0 - inward_factor)
            
        # Explosive launch
        elif phase < 0.3:
            local_phase = (phase - 0.2) / 0.1
            # Smooth extension from compressed state
            compression = self.compression_depth * ((1.0 - local_phase) ** 2)
            foot[2] = base_z + compression
            # Feet begin moving toward tuck position smoothly
            tuck_progress = (local_phase ** 2) * 0.2
            foot[0] = base_x * (0.95 - tuck_progress)
            foot[1] = base_y * (0.95 - tuck_progress)
            
        # Aerial rotation (tucked) - shortened phase
        elif phase < 0.65:
            local_phase = (phase - 0.3) / 0.35
            # Tuck peaks at mid-phase (apex) then begins releasing
            tuck_curve = np.sin(np.pi * local_phase)
            tuck_amount = self.tuck_factor * tuck_curve
            foot[0] = base_x * (0.9 - tuck_amount * 0.7)
            foot[1] = base_y * (0.9 - tuck_amount * 0.7)
            foot[2] = base_z + tuck_amount * 0.12
            
        # Landing preparation - extended phase for proper extension
        elif phase < 0.85:
            local_phase = (phase - 0.65) / 0.2
            # Smooth progressive extension to landing position
            extension_progress = local_phase ** 0.7
            # Feet extend downward and outward
            lateral_blend = np.sin(np.pi * extension_progress / 2.0)
            foot[0] = base_x * (0.9 + 0.1 * lateral_blend)
            foot[1] = base_y * (0.9 + 0.1 * lateral_blend)
            # Ensure feet reach below nominal stance for ground contact
            foot[2] = base_z - 0.02 * extension_progress
            
        # Impact and absorption
        elif phase < 0.95:
            local_phase = (phase - 0.85) / 0.1
            # Smooth compression during landing with cubic ease
            compression = self.compression_depth * np.sin(np.pi * local_phase / 2.0) ** 2
            foot[2] = base_z - 0.02 + compression
            # Feet at nominal lateral position
            foot[0] = base_x
            foot[1] = base_y
            
        # Stabilization
        else:
            local_phase = (phase - 0.95) / 0.05
            # Final smooth settling to nominal stance
            compression = self.compression_depth * ((1.0 - local_phase) ** 2)
            foot[2] = base_z + compression
            foot[0] = base_x
            foot[1] = base_y
        
        return foot