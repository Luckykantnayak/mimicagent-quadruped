from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising Spiral Jump: Vertical jump with continuous yaw rotation and sequential
    spiral leg extension pattern during flight.
    
    Phase breakdown:
      [0.0, 0.2]: compression - all legs compress, base lowers
      [0.2, 0.4]: launch - explosive extension, upward velocity, yaw begins
      [0.4, 0.6]: early_aerial_spiral - legs extend sequentially (FL->FR->RR->RL)
      [0.6, 0.8]: peak_spiral - maximum height and leg extension
      [0.8, 1.0]: descent_and_landing - legs retract, synchronized landing
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - tuned to respect safety constraints
        self.compression_depth = 0.06
        self.jump_height = 0.32
        self.total_yaw_rotation = 2.0 * np.pi
        
        # Reduced radial extension with asymmetric x-y scaling
        self.max_radial_extension_x = 1.16  # forward-back extension
        self.max_radial_extension_y = 1.10  # lateral extension (more conservative for hips)
        
        # Leg-specific extension limits (rear legs more conservative)
        self.leg_extension_factors = {
            leg_names[0]: 1.0,    # FL - full extension
            leg_names[1]: 1.0,    # FR - full extension
            leg_names[2]: 0.90,   # RR - reduced to protect hip
            leg_names[3]: 0.90,   # RL - reduced to protect hip
        }
        
        # Spiral sequence timing offsets for each leg
        self.spiral_phase_offsets = {
            leg_names[0]: 0.0,   # FL leads
            leg_names[1]: 0.05,  # FR follows
            leg_names[2]: 0.15,  # RR third
            leg_names[3]: 0.10,  # RL last
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
        Update base velocity commands based on phase.
        Velocities tuned to keep peak height below 0.68m envelope.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        if phase < 0.2:
            # Compression phase: gentle downward motion
            progress = phase / 0.2
            vz = -0.5 * (1.0 - progress)
            
        elif phase < 0.4:
            # Launch phase: upward velocity
            progress = (phase - 0.2) / 0.2
            vz = 1.9 * (1.0 - progress**0.5)
            yaw_rate = self.total_yaw_rotation / (0.8 * (1.0 / self.freq))
            
        elif phase < 0.8:
            # Aerial phase: parabolic trajectory
            progress = (phase - 0.4) / 0.4
            vz = 1.5 * (1.0 - 2.0 * progress)
            yaw_rate = self.total_yaw_rotation / (0.8 * (1.0 / self.freq))
            
        else:
            # Descent and landing phase
            progress = (phase - 0.8) / 0.2
            vz = -1.2 * (1.0 - 0.4 * progress)
            yaw_rate = self.total_yaw_rotation / (0.8 * (1.0 / self.freq)) * (1.0 - 0.5 * progress)
        
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
        Compute foot position in body frame with proper ground clearance
        and reduced radial extension to respect joint limits.
        
        Key fixes:
        - Minimal z-offsets during aerial phase (feet stay near nominal z)
        - Asymmetric x-y radial scaling to reduce hip stress
        - Leg-specific extension limits for rear legs
        - Smooth return to nominal during landing
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Get leg-specific extension factor
        leg_factor = self.leg_extension_factors[leg_name]
        spiral_offset = self.spiral_phase_offsets[leg_name]
        
        if phase < 0.2:
            # Compression: retract upward and inward
            progress = phase / 0.2
            compression_factor = np.sin(progress * np.pi / 2)
            # Increased compression for better launch preparation
            foot[0] *= (1.0 - 0.24 * compression_factor)
            foot[1] *= (1.0 - 0.24 * compression_factor)
            foot[2] += self.compression_depth * compression_factor
            
        elif phase < 0.4:
            # Launch: explosive downward extension
            progress = (phase - 0.2) / 0.2
            # Smoother extension curve to reduce joint stress
            extension_factor = progress**1.3
            foot[0] = base_pos[0] * (0.76 + 0.34 * extension_factor)
            foot[1] = base_pos[1] * (0.76 + 0.34 * extension_factor)
            foot[2] = base_pos[2] + self.compression_depth * (1.0 - extension_factor) - 0.03 * extension_factor
            
        elif phase < 0.75:
            # Aerial spiral phase: sequential radial extension
            aerial_phase = phase - 0.4
            leg_extension_start = spiral_offset
            leg_extension_duration = 0.20
            
            if aerial_phase < leg_extension_start:
                # Before this leg's extension starts
                foot[0] = base_pos[0] * 1.08
                foot[1] = base_pos[1] * 1.08
                # Keep z near nominal - no negative offset
                foot[2] = base_pos[2]
                
            elif aerial_phase < leg_extension_start + leg_extension_duration:
                # During extension: asymmetric radial outward motion
                ext_progress = (aerial_phase - leg_extension_start) / leg_extension_duration
                # Gentler extension curve
                ext_curve = ext_progress**1.4
                
                # Asymmetric scaling: more in x, less in y
                radial_mult_x = 1.08 + (self.max_radial_extension_x - 1.08) * ext_curve * leg_factor
                radial_mult_y = 1.08 + (self.max_radial_extension_y - 1.08) * ext_curve * leg_factor
                
                foot[0] = base_pos[0] * radial_mult_x
                foot[1] = base_pos[1] * radial_mult_y
                # Minimal z deviation - slight tuck during extension
                foot[2] = base_pos[2] + 0.01 * ext_curve
                
            else:
                # After peak extension: hold then begin retraction
                retract_start = leg_extension_start + leg_extension_duration
                retract_progress = (aerial_phase - retract_start) / (0.35 - retract_start)
                retract_progress = np.clip(retract_progress, 0.0, 1.0)
                
                if retract_progress < 0.4:
                    # Hold extended position briefly
                    foot[0] = base_pos[0] * (1.08 + (self.max_radial_extension_x - 1.08) * leg_factor)
                    foot[1] = base_pos[1] * (1.08 + (self.max_radial_extension_y - 1.08) * leg_factor)
                    foot[2] = base_pos[2] + 0.01
                else:
                    # Begin smooth retraction
                    ret_curve = (retract_progress - 0.4) / 0.6
                    ret_smooth = np.sin(ret_curve * np.pi / 2)
                    
                    radial_mult_x = (1.08 + (self.max_radial_extension_x - 1.08) * leg_factor) * (1.0 - ret_smooth) + 1.0 * ret_smooth
                    radial_mult_y = (1.08 + (self.max_radial_extension_y - 1.08) * leg_factor) * (1.0 - ret_smooth) + 1.0 * ret_smooth
                    
                    foot[0] = base_pos[0] * radial_mult_x
                    foot[1] = base_pos[1] * radial_mult_y
                    foot[2] = base_pos[2] + 0.01 * (1.0 - ret_smooth)
                    
        else:
            # Landing phase [0.75-1.0]: smooth return to nominal
            progress = (phase - 0.75) / 0.25
            land_curve = np.sin(progress * np.pi / 2)
            
            # Smooth interpolation back to nominal stance
            foot[0] = base_pos[0]
            foot[1] = base_pos[1]
            # Return to nominal z smoothly
            foot[2] = base_pos[2] * (1.0 + 0.02 * (1.0 - land_curve))
        
        return foot