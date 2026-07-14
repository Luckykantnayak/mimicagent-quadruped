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
        
        # Jump parameters
        self.compression_depth = 0.15  # How much base lowers during compression
        self.jump_height = 0.6  # Target peak height above initial position
        self.leg_compression_z = 0.12  # Leg compression in body frame during crouch
        
        # Spiral extension parameters
        self.spiral_extension_radius = 0.25  # Radial extension distance
        self.spiral_extension_z = 0.15  # Vertical extension component
        
        # Yaw rotation parameters
        self.total_yaw_rotation = 2 * np.pi  # 360 degrees total rotation
        self.yaw_rate_max = 6.0  # rad/s peak yaw rate
        
        # Velocity parameters
        self.launch_vz = 3.0  # Initial upward velocity during launch
        self.gravity = 9.81  # Kinematic gravity for velocity profile
        
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
        self.initial_z = 0.0  # Track initial height for jump trajectory

    def reset(self, root_pos, root_quat):
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.initial_z = root_pos[2]
        self.t = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Compression phase [0.0, 0.2]: downward velocity
        if phase < 0.2:
            local_phase = phase / 0.2
            vz = -self.compression_depth / (0.2 / self.freq)  # Constant downward velocity
        
        # Launch phase [0.2, 0.4]: strong upward velocity, yaw begins
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            vz = self.launch_vz * (1.0 - local_phase * 0.3)  # High upward velocity, slight decay
            yaw_rate = self.yaw_rate_max * local_phase  # Ramp up yaw rate
        
        # Aerial spiral extension [0.4, 0.6]: upward velocity decelerating
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            vz = self.launch_vz * (0.7 - local_phase * 1.2)  # Decelerate through zero
            yaw_rate = self.yaw_rate_max  # Constant yaw rate
        
        # Peak extension [0.6, 0.8]: transition to downward velocity
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            vz = -self.launch_vz * local_phase * 0.8  # Accelerate downward
            yaw_rate = self.yaw_rate_max  # Maintain yaw rate
        
        # Descent and landing [0.8, 1.0]: downward velocity, yaw stops
        else:
            local_phase = (phase - 0.8) / 0.2
            vz = -self.launch_vz * (0.8 + local_phase * 0.4)  # Continue downward
            yaw_rate = self.yaw_rate_max * (1.0 - local_phase)  # Decelerate yaw to zero
        
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
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine leg index for spiral direction
        leg_idx = self.leg_names.index(leg_name)
        
        # Spiral angle for this leg (evenly distributed around body)
        # FL: 45°, FR: -45°, RR: -135°, RL: 135°
        spiral_angles = {
            0: np.pi / 4,      # FL
            1: -np.pi / 4,     # FR
            2: -3 * np.pi / 4, # RR
            3: 3 * np.pi / 4,  # RL
        }
        spiral_angle = spiral_angles[leg_idx]
        
        # Compression phase [0.0, 0.2]: leg compresses, foot rises in body frame
        if phase < 0.2:
            local_phase = phase / 0.2
            compression_amount = self.leg_compression_z * np.sin(local_phase * np.pi / 2)
            foot[2] += compression_amount
        
        # Launch phase [0.2, 0.4]: transition from compressed to extended
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Decompress from compressed state
            compression_amount = self.leg_compression_z * (1.0 - local_phase)
            foot[2] += compression_amount
            # Begin slight outward extension
            extension_start = 0.1 * local_phase
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
            
            # Smooth extension using sine curve
            extension_factor = np.sin(spiral_phase * np.pi / 2)
            
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
            # Smooth retraction using cosine
            retraction_factor = 1.0 - np.sin(local_phase * np.pi / 2)
            
            # Retract from extended position back to base
            foot[0] += self.spiral_extension_radius * retraction_factor * np.cos(spiral_angle)
            foot[1] += self.spiral_extension_radius * retraction_factor * np.sin(spiral_angle)
            foot[2] += self.spiral_extension_z * retraction_factor
        
        return foot