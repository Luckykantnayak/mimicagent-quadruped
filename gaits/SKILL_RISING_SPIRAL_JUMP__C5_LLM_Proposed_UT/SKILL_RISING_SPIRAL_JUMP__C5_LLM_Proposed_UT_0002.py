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
        self.compression_depth = 0.06  # meters, moderate compression
        self.jump_height = 0.32  # meters, reduced to stay within height envelope
        self.max_radial_extension = 1.22  # reduced from 1.4 to respect joint limits
        self.total_yaw_rotation = 2.0 * np.pi  # 360 degrees total rotation
        
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
        Velocities reduced to keep peak height below 0.68m envelope.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        if phase < 0.2:
            # Compression phase: slight downward motion
            progress = phase / 0.2
            vz = -0.5 * (1.0 - progress)  # gentle downward velocity
            
        elif phase < 0.4:
            # Launch phase: upward velocity reduced to 1.9 m/s peak
            progress = (phase - 0.2) / 0.2
            vz = 1.9 * (1.0 - progress**0.5)  # reduced from 2.5
            # Initiate yaw rotation
            yaw_rate = self.total_yaw_rotation / (0.8 * (1.0 / self.freq))
            
        elif phase < 0.8:
            # Aerial phase: parabolic trajectory with reduced velocity range
            progress = (phase - 0.4) / 0.4
            vz = 1.5 * (1.0 - 2.0 * progress)  # reduced from 2.0, range now +1.5 to -1.5
            yaw_rate = self.total_yaw_rotation / (0.8 * (1.0 / self.freq))
            
        else:
            # Descent and landing phase
            progress = (phase - 0.8) / 0.2
            vz = -1.2 * (1.0 - 0.4 * progress)  # controlled descent, slowing for landing
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
        Compute foot position in body frame with improved ground clearance
        and reduced radial extension to respect joint limits.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine leg-specific spiral timing
        spiral_offset = self.spiral_phase_offsets[leg_name]
        
        if phase < 0.2:
            # Compression: retract upward and inward (reduced inward factor)
            progress = phase / 0.2
            compression_factor = np.sin(progress * np.pi / 2)
            foot[0] *= (1.0 - 0.18 * compression_factor)  # reduced from 0.3 to 0.18
            foot[1] *= (1.0 - 0.18 * compression_factor)
            foot[2] += self.compression_depth * compression_factor
            
        elif phase < 0.4:
            # Launch: explosive downward extension
            progress = (phase - 0.2) / 0.2
            extension_factor = np.sin(progress * np.pi / 2)
            foot[0] = base_pos[0] * (0.82 + 0.28 * extension_factor)  # smooth to 1.1x
            foot[1] = base_pos[1] * (0.82 + 0.28 * extension_factor)
            foot[2] = base_pos[2] + self.compression_depth * (1.0 - extension_factor) - 0.04 * extension_factor
            
        elif phase < 0.75:
            # Aerial spiral phase: sequential radial extension with maintained ground clearance
            aerial_phase = phase - 0.4
            leg_extension_start = spiral_offset
            leg_extension_duration = 0.22
            
            if aerial_phase < leg_extension_start:
                # Before this leg's extension starts
                foot[0] = base_pos[0] * 1.1
                foot[1] = base_pos[1] * 1.1
                foot[2] = base_pos[2] - 0.06  # maintain clearance, no upward offset
                
            elif aerial_phase < leg_extension_start + leg_extension_duration:
                # During extension: radial outward motion
                ext_progress = (aerial_phase - leg_extension_start) / leg_extension_duration
                ext_curve = np.sin(ext_progress * np.pi / 2)
                radial_mult = 1.1 + (self.max_radial_extension - 1.1) * ext_curve
                foot[0] = base_pos[0] * radial_mult
                foot[1] = base_pos[1] * radial_mult
                foot[2] = base_pos[2] - 0.06  # keep feet low, no upward lift
                
            else:
                # After peak extension: hold briefly then begin retraction
                retract_start = leg_extension_start + leg_extension_duration
                retract_progress = (aerial_phase - retract_start) / (0.35 - retract_start)
                retract_progress = np.clip(retract_progress, 0.0, 1.0)
                
                # Smooth retraction from extended to nominal
                ret_curve = np.sin(retract_progress * np.pi / 2)
                radial_mult = self.max_radial_extension - (self.max_radial_extension - 1.0) * ret_curve
                foot[0] = base_pos[0] * radial_mult
                foot[1] = base_pos[1] * radial_mult
                foot[2] = base_pos[2] - 0.06 - 0.04 * ret_curve  # drive downward during retraction
                    
        else:
            # Landing phase [0.75-1.0]: extended retraction window for safe landing
            progress = (phase - 0.75) / 0.25
            land_curve = np.sin(progress * np.pi / 2)
            
            # Smooth return to nominal with aggressive downward positioning
            current_radial = 1.0 + (self.max_radial_extension - 1.0) * (1.0 - land_curve)
            foot[0] = base_pos[0] * current_radial
            foot[1] = base_pos[1] * current_radial
            # Ensure feet are well below base during descent
            foot[2] = base_pos[2] - 0.10 * (1.0 + 0.5 * land_curve)
        
        return foot