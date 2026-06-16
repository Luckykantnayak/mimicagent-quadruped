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
        self.compression_depth = 0.10  # 10cm vertical compression
        
        # Launch parameters
        self.launch_vz = 2.2  # Upward velocity at launch (m/s)
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

    def update_base_motion(self, phase, dt):
        """
        Update base position and orientation based on phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        if phase < 0.2:
            # Compression phase: slight downward velocity as legs compress
            compression_progress = phase / 0.2
            vz = -0.5 * (1.0 - compression_progress)  # Decelerate to zero
            
        elif phase < 0.4:
            # Launch phase: explosive upward velocity, yaw rotation starts
            launch_progress = (phase - 0.2) / 0.2
            vz = self.launch_vz * launch_progress  # Ramp up to launch velocity
            yaw_rate = self.yaw_rate_max * launch_progress  # Ramp up yaw rate
            
        elif phase < 0.8:
            # Aerial phase: ballistic trajectory with gravity, sustained yaw rotation
            aerial_progress = (phase - 0.4) / 0.4
            time_since_launch = aerial_progress * (0.4 / self.freq)
            
            # Ballistic vertical velocity: v(t) = v0 - g*t
            vz = self.launch_vz - self.gravity * time_since_launch
            
            # Sustained yaw rotation
            yaw_rate = self.yaw_rate_max
            
        else:
            # Descent and landing: downward velocity decreasing, yaw rate decreasing
            landing_progress = (phase - 0.8) / 0.2
            
            # Decelerate downward velocity toward zero
            vz = -1.0 * (1.0 - landing_progress)
            
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

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase and leg-specific spiral timing.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg-specific parameters
        nominal_radius = np.sqrt(foot[0]**2 + foot[1]**2)
        nominal_angle = np.arctan2(foot[1], foot[0])
        
        if phase < 0.2:
            # Compression: foot moves upward (leg compresses)
            compression_progress = phase / 0.2
            foot[2] += self.compression_depth * compression_progress
            
        elif phase < 0.4:
            # Launch: foot extends downward (leg extends)
            launch_progress = (phase - 0.2) / 0.2
            foot[2] += self.compression_depth * (1.0 - launch_progress)
            
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
            
            # Slight upward lift during extension for visual effect
            foot[2] += 0.05 * extension_factor
            
        else:
            # Descent and landing: retract legs toward nominal stance
            landing_progress = (phase - 0.8) / 0.2
            
            # Smooth retraction using sinusoidal profile
            retraction_factor = 0.5 * (1.0 - np.cos(np.pi * landing_progress))
            
            # Interpolate from extended position back to nominal
            # Compute extended radius (from previous phase)
            extended_radius = nominal_radius + self.radial_extension_max
            current_radius = extended_radius - self.radial_extension_max * retraction_factor
            
            foot[0] = current_radius * np.cos(nominal_angle)
            foot[1] = current_radius * np.sin(nominal_angle)
            
            # Lower foot back to nominal height
            foot[2] = self.base_feet_pos_body[leg_name][2] + 0.05 * (1.0 - retraction_factor)
        
        return foot