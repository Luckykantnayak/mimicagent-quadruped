from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising Spiral Jump: vertical jump with yaw rotation and sequential spiral leg extension.
    
    Phase breakdown:
    - [0.0, 0.2]: Compression - all legs compress synchronously
    - [0.2, 0.4]: Launch - explosive extension, upward velocity, yaw rotation starts
    - [0.4, 0.8]: Aerial spiral - ballistic flight with sequential leg extension (FL→FR→RR→RL)
    - [0.8, 1.0]: Descent and landing - legs retract, contact re-established
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for complete jump cycle
        
        # Base foot positions (nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compression parameters
        self.compression_depth = 0.08  # Base descends 8cm during crouch
        
        # Launch parameters
        self.launch_vz = 1.6  # Upward velocity at launch (m/s), tuned for safe envelope
        self.gravity = 9.81  # Gravitational acceleration
        
        # Yaw rotation parameters
        self.total_yaw_rotation = np.pi  # 180 degrees total rotation
        self.yaw_rate_max = 3.0  # rad/s during aerial phase
        
        # Spiral leg extension parameters
        self.radial_extension_max = 0.15  # Maximum outward extension from nominal
        
        # Spiral phase offsets for sequential extension (FL→FR→RR→RL)
        self.spiral_phase_offsets = {
            leg_names[0]: 0.0,   # FL leads
            leg_names[1]: 0.1,   # FR follows
            leg_names[2]: 0.15,  # RR third
            leg_names[3]: 0.2,   # RL last
        }
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Height envelope constraints
        self.z_min = -0.12  # Minimum safe base height
        self.z_max = 0.75   # Maximum safe base height
        self.nominal_base_height = 0.0  # Target landing height

    def update_base_motion(self, phase, dt):
        """
        Update base position and orientation based on phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        if phase < 0.2:
            # Compression phase: controlled descent for crouch
            compression_progress = phase / 0.2
            # Smooth descent profile, decelerate toward end of compression
            compression_speed = 0.4 * (1.0 - compression_progress)
            vz = -compression_speed
            
        elif phase < 0.4:
            # Launch phase: explosive upward velocity, yaw rotation starts
            launch_progress = (phase - 0.2) / 0.2
            # Smooth launch acceleration using sinusoidal ramp
            vz = self.launch_vz * np.sin(launch_progress * np.pi * 0.5)
            yaw_rate = self.yaw_rate_max * launch_progress
            
        elif phase < 0.8:
            # Aerial phase: ballistic trajectory with gravity, sustained yaw rotation
            aerial_progress = (phase - 0.4) / 0.4
            time_since_launch = aerial_progress * (0.4 / self.freq)
            
            # Ballistic vertical velocity: v(t) = v0 - g*t
            vz = self.launch_vz - self.gravity * time_since_launch
            
            # Sustained yaw rotation
            yaw_rate = self.yaw_rate_max
            
        else:
            # Descent and landing: controlled descent to target height
            landing_progress = (phase - 0.8) / 0.2
            
            # Decelerate downward velocity smoothly to zero
            landing_speed = 0.8 * (1.0 - landing_progress)
            vz = -landing_speed
            
            # Decelerate yaw rate to zero
            yaw_rate = self.yaw_rate_max * (1.0 - landing_progress)
        
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
        
        # Enforce height envelope constraints
        if self.root_pos[2] < self.z_min:
            self.root_pos[2] = self.z_min
            if vz < 0:
                self.vel_world[2] = 0.0
        elif self.root_pos[2] > self.z_max:
            self.root_pos[2] = self.z_max
            if vz > 0:
                self.vel_world[2] = 0.0

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase and leg-specific spiral timing.
        Feet remain grounded during compression and landing, lift during aerial phase.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg-specific parameters
        nominal_radius = np.sqrt(foot[0]**2 + foot[1]**2)
        nominal_angle = np.arctan2(foot[1], foot[0])
        nominal_z = foot[2]
        
        if phase < 0.2:
            # Compression: feet stay grounded (no z change in body frame)
            # Legs compress as base descends while feet remain at nominal height
            pass
            
        elif phase < 0.4:
            # Launch: feet push off, slight lift at end of launch
            launch_progress = (phase - 0.2) / 0.2
            # Small lift as legs extend for launch
            lift_amount = 0.02 * np.sin(launch_progress * np.pi)
            foot[2] = nominal_z + lift_amount
            
        elif phase < 0.8:
            # Aerial spiral expansion phase
            aerial_phase = (phase - 0.4) / 0.4
            
            # Get leg-specific spiral offset
            spiral_offset = self.spiral_phase_offsets.get(leg_name, 0.0)
            
            # Compute spiral progress for this leg
            spiral_progress = np.clip((aerial_phase - spiral_offset) / 0.25, 0.0, 1.0)
            
            # Smooth extension using sinusoidal profile
            extension_factor = 0.5 * (1.0 - np.cos(np.pi * spiral_progress))
            
            # Radial extension outward
            radial_extension = self.radial_extension_max * extension_factor
            new_radius = nominal_radius + radial_extension
            
            # Update foot position with extended radius
            foot[0] = new_radius * np.cos(nominal_angle)
            foot[1] = new_radius * np.sin(nominal_angle)
            
            # Upward lift during aerial phase for visual spiral effect
            foot[2] = nominal_z + 0.04 * extension_factor
            
        else:
            # Descent and landing: retract legs toward nominal stance, prepare for ground contact
            landing_progress = (phase - 0.8) / 0.2
            
            # Smooth retraction using sinusoidal profile
            retraction_factor = 0.5 * (1.0 + np.cos(np.pi * (1.0 - landing_progress)))
            
            # Interpolate from extended position back to nominal
            extended_radius = nominal_radius + self.radial_extension_max
            current_radius = nominal_radius + self.radial_extension_max * retraction_factor
            
            foot[0] = current_radius * np.cos(nominal_angle)
            foot[1] = current_radius * np.sin(nominal_angle)
            
            # Lower foot to prepare for ground contact
            # Gradually return to nominal z and then extend slightly below for landing
            aerial_z_offset = 0.04 * retraction_factor
            landing_extension = -0.01 * (1.0 - retraction_factor)  # Extend down for landing
            foot[2] = nominal_z + aerial_z_offset + landing_extension
        
        return foot