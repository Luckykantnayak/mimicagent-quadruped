from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising Spiral Jump: Vertical jump with continuous yaw rotation and sequential
    spiral leg extension pattern during flight.
    
    Phase breakdown:
      [0.0, 0.2]: Compression - all legs compress, base lowers
      [0.2, 0.4]: Launch - explosive extension, base rises, yaw begins
      [0.4, 0.6]: Aerial spiral extension - legs extend sequentially (FL→FR→RR→RL)
      [0.6, 0.8]: Peak extension - maximum height, all legs extended, yaw continues
      [0.8, 1.0]: Descent and landing - legs retract, yaw stops, simultaneous touchdown
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for dramatic jump
        
        # Base foot positions (nominal stance in body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Jump parameters - REDUCED to respect height envelope [0.1, 0.68]
        self.compression_depth = 0.10  # Reduced from 0.15 to minimize total height range
        self.launch_vz = 1.2  # Reduced from 3.0 to constrain peak height within limits
        
        # Leg compression during crouch
        self.leg_compression_z = 0.10  # Reduced from 0.12 for safety
        
        # Spiral extension parameters - kept moderate
        self.spiral_extension_radius = 0.22  # Slightly reduced from 0.25
        self.spiral_extension_z = 0.12  # Reduced from 0.15 to limit leg reach
        
        # Yaw rotation parameters
        self.total_yaw_rotation = 2 * np.pi  # 360 degrees total rotation
        self.yaw_rate_max = 6.0  # rad/s peak yaw rate
        
        # Spiral sequence timing offsets (FL→FR→RR→RL)
        self.spiral_phase_offsets = {
            leg_names[0]: 0.0,   # FL first
            leg_names[1]: 0.25,  # FR second
            leg_names[2]: 0.5,   # RR third
            leg_names[3]: 0.75,  # RL fourth
        }
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.initial_z = 0.0

    def reset(self, root_pos, root_quat):
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.initial_z = root_pos[2]
        self.t = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Velocity profile tuned to keep base height within [0.1, 0.68]m envelope.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Compression phase [0.0, 0.2]: gentle downward velocity
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth compression using sine curve
            vz = -self.compression_depth / (0.2 / self.freq) * np.sin(local_phase * np.pi)
        
        # Launch phase [0.2, 0.4]: upward velocity with smooth ramp-up, yaw begins
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Smooth launch with sine curve to reduce jerk
            vz = self.launch_vz * np.sin(local_phase * np.pi / 2)
            # Ramp up yaw rate smoothly
            yaw_rate = self.yaw_rate_max * (local_phase ** 2)
        
        # Aerial spiral extension [0.4, 0.6]: upward velocity decelerating
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Decelerate upward velocity smoothly through peak
            vz = self.launch_vz * np.cos(local_phase * np.pi)
            yaw_rate = self.yaw_rate_max  # Constant yaw rate
        
        # Peak extension [0.6, 0.8]: transition to downward velocity
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Accelerate downward smoothly
            vz = -self.launch_vz * np.sin(local_phase * np.pi / 2) * 0.9
            yaw_rate = self.yaw_rate_max  # Maintain yaw rate
        
        # Descent and landing [0.8, 1.0]: downward velocity, yaw stops
        else:
            local_phase = (phase - 0.8) / 0.2
            # Continue downward with deceleration near landing
            vz = -self.launch_vz * 0.9 * (1.0 - local_phase * 0.5)
            # Smoothly decelerate yaw to zero
            yaw_rate = self.yaw_rate_max * np.cos(local_phase * np.pi / 2)
        
        # Set velocity commands
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
        Compute foot position in body frame based on phase and leg-specific spiral timing.
        Uses smooth transitions to minimize jerk and maintain joint limits.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine leg index for spiral direction
        leg_idx = self.leg_names.index(leg_name)
        
        # Spiral angle for this leg (evenly distributed around body)
        spiral_angles = {
            0: np.pi / 4,      # FL: 45°
            1: -np.pi / 4,     # FR: -45°
            2: -3 * np.pi / 4, # RR: -135°
            3: 3 * np.pi / 4,  # RL: 135°
        }
        spiral_angle = spiral_angles[leg_idx]
        
        # Compression phase [0.0, 0.2]: leg compresses, foot rises in body frame
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth compression using sine
            compression_amount = self.leg_compression_z * np.sin(local_phase * np.pi / 2)
            foot[2] += compression_amount
        
        # Launch phase [0.2, 0.4]: transition from compressed to extended
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Smooth decompression
            compression_amount = self.leg_compression_z * np.cos(local_phase * np.pi / 2)
            foot[2] += compression_amount
            # Begin slight outward extension with smooth curve
            extension_start = 0.08 * np.sin(local_phase * np.pi / 2)
            foot[0] += extension_start * np.cos(spiral_angle)
            foot[1] += extension_start * np.sin(spiral_angle)
        
        # Aerial spiral extension [0.4, 0.6]: sequential spiral extension
        elif phase < 0.6:
            # Get spiral-specific phase offset
            spiral_offset = self.spiral_phase_offsets[leg_name]
            # Map phase [0.4, 0.6] to local [0, 1] for spiral timing
            local_phase = (phase - 0.4) / 0.2
            # Apply offset and clamp
            spiral_phase = np.clip(local_phase - spiral_offset, 0.0, 1.0)
            
            # Smooth extension using smoothstep for reduced jerk
            extension_factor = spiral_phase * spiral_phase * (3.0 - 2.0 * spiral_phase)
            
            # Radial extension in spiral direction
            foot[0] += self.spiral_extension_radius * extension_factor * np.cos(spiral_angle)
            foot[1] += self.spiral_extension_radius * extension_factor * np.sin(spiral_angle)
            foot[2] += self.spiral_extension_z * extension_factor
        
        # Peak extension [0.6, 0.8]: hold maximum extension
        elif phase < 0.8:
            # Full extension maintained
            foot[0] += self.spiral_extension_radius * np.cos(spiral_angle)
            foot[1] += self.spiral_extension_radius * np.sin(spiral_angle)
            foot[2] += self.spiral_extension_z
        
        # Descent and landing [0.8, 1.0]: retract to nominal stance
        else:
            local_phase = (phase - 0.8) / 0.2
            # Smooth retraction using smoothstep
            retraction_progress = local_phase * local_phase * (3.0 - 2.0 * local_phase)
            retraction_factor = 1.0 - retraction_progress
            
            # Retract from extended position back to base
            foot[0] += self.spiral_extension_radius * retraction_factor * np.cos(spiral_angle)
            foot[1] += self.spiral_extension_radius * retraction_factor * np.sin(spiral_angle)
            foot[2] += self.spiral_extension_z * retraction_factor
        
        return foot