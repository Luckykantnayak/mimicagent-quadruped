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
        
        # Jump height and trajectory parameters
        self.peak_height = 0.35  # Peak height above nominal (well within z_max=0.75)
        
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

    def compute_base_height_profile(self, phase):
        """
        Compute target base height as smooth function of phase.
        This ensures height stays within bounds by construction.
        """
        if phase < 0.2:
            # Compression: smooth descent to -compression_depth
            compression_progress = phase / 0.2
            # Smooth interpolation using cosine
            t = 0.5 * (1.0 - np.cos(np.pi * compression_progress))
            target_height = -self.compression_depth * t
            
        elif phase < 0.4:
            # Launch: rapid ascent from compressed to start of aerial
            launch_progress = (phase - 0.2) / 0.2
            t = 0.5 * (1.0 - np.cos(np.pi * launch_progress))
            # Interpolate from bottom of compression to start of parabolic arc
            target_height = -self.compression_depth * (1.0 - t) + 0.05 * t
            
        elif phase < 0.8:
            # Aerial: smooth parabolic arc peaking at phase 0.6
            aerial_progress = (phase - 0.4) / 0.4
            # Sinusoidal arc: peaks at aerial_progress=0.5 (phase=0.6)
            target_height = 0.05 + self.peak_height * np.sin(np.pi * aerial_progress)
            
        else:
            # Descent and landing: smooth return to nominal
            landing_progress = (phase - 0.8) / 0.2
            t = 0.5 * (1.0 - np.cos(np.pi * landing_progress))
            # From end of aerial arc (near ground) to nominal
            aerial_end_height = 0.05 + self.peak_height * np.sin(np.pi)  # ~0.05
            target_height = aerial_end_height * (1.0 - t) + self.nominal_base_height * t
        
        return target_height

    def update_base_motion(self, phase, dt):
        """
        Update base position and orientation based on phase.
        Velocity is derived from height profile to ensure smooth bounded motion.
        """
        # Compute current and next target height
        current_target_height = self.compute_base_height_profile(phase)
        
        # Estimate velocity by finite difference of height profile
        phase_next = phase + self.freq * dt
        if phase_next > 1.0:
            phase_next = 1.0
        next_target_height = self.compute_base_height_profile(phase_next)
        
        if dt > 0:
            vz = (next_target_height - current_target_height) / dt
        else:
            vz = 0.0
        
        # Compute yaw rate
        yaw_rate = 0.0
        if phase < 0.2:
            # Compression: no rotation
            yaw_rate = 0.0
        elif phase < 0.4:
            # Launch: yaw rotation ramps up
            launch_progress = (phase - 0.2) / 0.2
            yaw_rate = self.yaw_rate_max * launch_progress
        elif phase < 0.8:
            # Aerial: sustained yaw rotation
            yaw_rate = self.yaw_rate_max
        else:
            # Landing: decelerate yaw
            landing_progress = (phase - 0.8) / 0.2
            yaw_rate = self.yaw_rate_max * (1.0 - landing_progress)
        
        # Set velocity commands
        self.vel_world = np.array([0.0, 0.0, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Apply soft clamping to stay within envelope
        if self.root_pos[2] < self.z_min:
            self.root_pos[2] = self.z_min
            if self.vel_world[2] < 0:
                self.vel_world[2] = 0.0
        elif self.root_pos[2] > self.z_max:
            self.root_pos[2] = self.z_max
            if self.vel_world[2] > 0:
                self.vel_world[2] = 0.0

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase and leg-specific spiral timing.
        Accounts for base motion to maintain ground contact during compression/landing.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg-specific parameters
        nominal_radius = np.sqrt(foot[0]**2 + foot[1]**2)
        nominal_angle = np.arctan2(foot[1], foot[0])
        nominal_z = foot[2]
        
        # Compute base height offset to compensate for base motion during ground contact
        base_height = self.compute_base_height_profile(phase)
        
        if phase < 0.2:
            # Compression: feet stay on ground (z_world = 0)
            # As base descends, foot z in body frame must increase to maintain ground contact
            # foot_z_body = nominal_z - base_height (to cancel base descent)
            foot[2] = nominal_z - base_height
            
        elif phase < 0.4:
            # Launch: transition from ground-locked to airborne
            launch_progress = (phase - 0.2) / 0.2
            # Gradually reduce ground compensation and add small lift
            compression_compensation = -(-self.compression_depth) * (1.0 - launch_progress)
            lift_amount = 0.03 * launch_progress
            foot[2] = nominal_z + compression_compensation + lift_amount
            
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
            foot[2] = nominal_z + 0.05 * extension_factor
            
        else:
            # Descent and landing: retract legs toward nominal stance
            landing_progress = (phase - 0.8) / 0.2
            
            # Smooth retraction using sinusoidal profile
            retraction_factor = 0.5 * (1.0 + np.cos(np.pi * (1.0 - landing_progress)))
            
            # Interpolate from extended position back to nominal
            current_radius = nominal_radius + self.radial_extension_max * retraction_factor
            
            foot[0] = current_radius * np.cos(nominal_angle)
            foot[1] = current_radius * np.sin(nominal_angle)
            
            # Transition from aerial z back to ground-contact configuration
            aerial_z_offset = 0.05 * retraction_factor
            # As landing progresses, compensate for base height to prepare for ground contact
            ground_compensation = -base_height * landing_progress
            foot[2] = nominal_z + aerial_z_offset + ground_compensation
        
        return foot