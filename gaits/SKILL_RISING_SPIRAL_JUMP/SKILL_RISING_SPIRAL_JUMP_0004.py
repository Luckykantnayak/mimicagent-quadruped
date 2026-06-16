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
        
        # Compute nominal standing height from foot positions
        # When feet are at their nominal body positions and base is at this height, feet touch ground
        min_foot_z = min(foot[2] for foot in initial_foot_positions_body.values())
        self.nominal_standing_height = -min_foot_z  # Base height above ground in standing pose
        
        # Compression parameters
        self.compression_depth = 0.08  # Base descends 8cm during crouch
        
        # Jump height and trajectory parameters
        self.peak_height = 0.35  # Peak height above nominal standing height
        
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
        
        # Base state - initialize at nominal standing height
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.nominal_standing_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Height envelope constraints (absolute world coordinates)
        self.z_min = -0.12  # Minimum safe base height
        self.z_max = 0.75   # Maximum safe base height

    def compute_base_height_profile(self, phase):
        """
        Compute absolute world z-coordinate for base as smooth function of phase.
        All heights are in world frame with ground at z=0.
        Returns height guaranteed to be within [z_min, z_max].
        """
        if phase < 0.2:
            # Compression: smooth descent from standing to compressed
            compression_progress = phase / 0.2
            t = 0.5 * (1.0 - np.cos(np.pi * compression_progress))
            target_height = self.nominal_standing_height - self.compression_depth * t
            
        elif phase < 0.4:
            # Launch: rapid ascent from compressed to start of aerial arc
            launch_progress = (phase - 0.2) / 0.2
            t = 0.5 * (1.0 - np.cos(np.pi * launch_progress))
            compressed_height = self.nominal_standing_height - self.compression_depth
            aerial_start_height = self.nominal_standing_height + 0.05
            target_height = compressed_height * (1.0 - t) + aerial_start_height * t
            
        elif phase < 0.8:
            # Aerial: smooth parabolic arc peaking at phase 0.6
            aerial_progress = (phase - 0.4) / 0.4
            # Sinusoidal arc: peaks at aerial_progress=0.5 (phase=0.6)
            arc_offset = self.peak_height * np.sin(np.pi * aerial_progress)
            target_height = self.nominal_standing_height + 0.05 + arc_offset
            
        else:
            # Descent and landing: smooth return to nominal standing height
            landing_progress = (phase - 0.8) / 0.2
            t = 0.5 * (1.0 - np.cos(np.pi * landing_progress))
            # From end of aerial arc back to standing
            aerial_end_height = self.nominal_standing_height + 0.05 + self.peak_height * np.sin(np.pi)
            target_height = aerial_end_height * (1.0 - t) + self.nominal_standing_height * t
        
        # Clamp to envelope bounds for safety
        target_height = np.clip(target_height, self.z_min, self.z_max)
        
        return target_height

    def update_base_motion(self, phase, dt):
        """
        Update base position and orientation based on phase.
        Base height is directly assigned from profile to eliminate drift.
        """
        # Compute target height for current phase
        target_height = self.compute_base_height_profile(phase)
        
        # Store previous height for velocity estimation
        previous_height = self.root_pos[2]
        
        # Directly assign target height (no integration for z-axis)
        self.root_pos[2] = target_height
        
        # Estimate vertical velocity for continuity
        if dt > 0:
            vz = (self.root_pos[2] - previous_height) / dt
        else:
            vz = 0.0
        
        # Compute yaw rate based on phase
        yaw_rate = 0.0
        if phase < 0.2:
            # Compression: no rotation
            yaw_rate = 0.0
        elif phase < 0.4:
            # Launch: yaw rotation ramps up smoothly
            launch_progress = (phase - 0.2) / 0.2
            ramp = 0.5 * (1.0 - np.cos(np.pi * launch_progress))
            yaw_rate = self.yaw_rate_max * ramp
        elif phase < 0.8:
            # Aerial: sustained yaw rotation
            yaw_rate = self.yaw_rate_max
        else:
            # Landing: decelerate yaw smoothly
            landing_progress = (phase - 0.8) / 0.2
            decel = 0.5 * (1.0 + np.cos(np.pi * landing_progress))
            yaw_rate = self.yaw_rate_max * decel
        
        # Set velocity commands
        self.vel_world = np.array([0.0, 0.0, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate orientation only (position x,y remain at origin, z is directly assigned)
        # Update quaternion based on angular velocity
        if dt > 0:
            delta_quat = quat_from_euler_xyz(self.omega_world * dt)
            self.root_quat = quat_mul(self.root_quat, delta_quat)
            self.root_quat = self.root_quat / np.linalg.norm(self.root_quat)

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase and leg-specific spiral timing.
        During ground contact phases, feet extend in body frame to maintain ground contact.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg-specific parameters
        nominal_radius = np.sqrt(foot[0]**2 + foot[1]**2)
        nominal_angle = np.arctan2(foot[1], foot[0])
        nominal_z = foot[2]
        
        # Get current base height
        base_height = self.compute_base_height_profile(phase)
        
        if phase < 0.2:
            # Compression: feet stay on ground (z_world = 0)
            # foot_z_world = base_height + foot_z_body = 0
            # Therefore: foot_z_body = -base_height
            foot[2] = -base_height
            
        elif phase < 0.4:
            # Launch: transition from ground-locked to airborne
            launch_progress = (phase - 0.2) / 0.2
            t = 0.5 * (1.0 - np.cos(np.pi * launch_progress))
            
            # Interpolate from ground contact to slight lift
            compressed_base_height = self.nominal_standing_height - self.compression_depth
            ground_contact_z = -compressed_base_height
            aerial_start_z = nominal_z + 0.03
            
            foot[2] = ground_contact_z * (1.0 - t) + aerial_start_z * t
            
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
            # Descent and landing: retract legs toward ground contact
            landing_progress = (phase - 0.8) / 0.2
            t = 0.5 * (1.0 - np.cos(np.pi * landing_progress))
            
            # Retract radius from extended back to nominal
            retraction_factor = 1.0 - t
            current_radius = nominal_radius + self.radial_extension_max * retraction_factor
            
            foot[0] = current_radius * np.cos(nominal_angle)
            foot[1] = current_radius * np.sin(nominal_angle)
            
            # Transition from aerial z back to ground-contact configuration
            aerial_z = nominal_z + 0.05 * retraction_factor
            ground_contact_z = -base_height
            
            foot[2] = aerial_z * (1.0 - t) + ground_contact_z * t
        
        return foot