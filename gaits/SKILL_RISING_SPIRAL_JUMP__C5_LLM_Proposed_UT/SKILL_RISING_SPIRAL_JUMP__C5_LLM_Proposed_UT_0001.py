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
        
        # Motion parameters
        self.compression_depth = 0.08  # meters, vertical compression during crouch
        self.jump_height = 0.4  # meters, peak height above nominal
        self.max_radial_extension = 1.4  # multiplier for spiral leg extension
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
        
        Compression [0.0-0.2]: slight downward velocity
        Launch [0.2-0.4]: strong upward velocity, yaw begins
        Aerial [0.4-0.8]: parabolic trajectory (vz decreases through zero)
        Descent [0.8-1.0]: downward velocity, yaw continues
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        if phase < 0.2:
            # Compression phase: slight downward motion
            progress = phase / 0.2
            vz = -0.4 * (1.0 - progress)  # decreasing downward velocity
            
        elif phase < 0.4:
            # Launch phase: explosive upward velocity, yaw starts
            progress = (phase - 0.2) / 0.2
            # Strong upward impulse
            vz = 2.5 * (1.0 - progress**0.5)  # decreasing as we gain height
            # Initiate yaw rotation
            yaw_rate = self.total_yaw_rotation / (0.8 * (1.0 / self.freq))  # constant rate from 0.2 to 1.0
            
        elif phase < 0.8:
            # Aerial phase: parabolic trajectory
            progress = (phase - 0.4) / 0.4
            # Velocity transitions from positive (rising) to negative (falling)
            # Peak occurs around phase 0.6 (progress = 0.5)
            vz = 2.0 * (1.0 - 2.0 * progress)  # linear decrease from +2 to -2
            yaw_rate = self.total_yaw_rotation / (0.8 * (1.0 / self.freq))
            
        else:
            # Descent and landing phase
            progress = (phase - 0.8) / 0.2
            vz = -1.5 * (1.0 - 0.3 * progress)  # downward, slowing for landing
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
        Compute foot position in body frame based on phase and leg-specific spiral timing.
        
        Each leg follows:
          [0.0-0.2]: compression (retract upward/inward)
          [0.2-0.4]: launch extension (push downward)
          [0.4-0.7/0.85]: spiral extension (outward radial motion, sequential)
          [0.7/0.85-1.0]: retraction for landing
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine leg-specific spiral timing
        spiral_offset = self.spiral_phase_offsets[leg_name]
        
        if phase < 0.2:
            # Compression: retract upward and inward
            progress = phase / 0.2
            compression_factor = np.sin(progress * np.pi / 2)  # smooth 0->1
            foot[0] *= (1.0 - 0.3 * compression_factor)  # move inward in x
            foot[1] *= (1.0 - 0.3 * compression_factor)  # move inward in y
            foot[2] += self.compression_depth * compression_factor  # move upward (less negative z)
            
        elif phase < 0.4:
            # Launch: explosive downward extension
            progress = (phase - 0.2) / 0.2
            extension_factor = np.sin(progress * np.pi / 2)
            # Return toward nominal, then extend slightly beyond
            foot[0] = base_pos[0] * (0.7 + 0.4 * extension_factor)
            foot[1] = base_pos[1] * (0.7 + 0.4 * extension_factor)
            foot[2] = base_pos[2] + self.compression_depth * (1.0 - extension_factor) - 0.05 * extension_factor
            
        elif phase < 0.85:
            # Aerial spiral phase: sequential radial extension
            # Each leg has its own extension window based on spiral_offset
            aerial_phase = phase - 0.4
            leg_extension_start = spiral_offset
            leg_extension_duration = 0.25
            
            if aerial_phase < leg_extension_start:
                # Before this leg's extension starts: maintain launch position
                foot[0] = base_pos[0] * 1.1
                foot[1] = base_pos[1] * 1.1
                foot[2] = base_pos[2] - 0.05
                
            elif aerial_phase < leg_extension_start + leg_extension_duration:
                # During extension: radial outward motion
                ext_progress = (aerial_phase - leg_extension_start) / leg_extension_duration
                ext_curve = np.sin(ext_progress * np.pi / 2)  # smooth acceleration
                radial_mult = 1.1 + (self.max_radial_extension - 1.1) * ext_curve
                foot[0] = base_pos[0] * radial_mult
                foot[1] = base_pos[1] * radial_mult
                foot[2] = base_pos[2] - 0.05 + 0.1 * ext_curve  # slight upward component
                
            else:
                # After peak extension: hold extended or begin retraction
                retract_start = leg_extension_start + leg_extension_duration
                retract_progress = (aerial_phase - retract_start) / (0.45 - retract_start)
                retract_progress = np.clip(retract_progress, 0.0, 1.0)
                
                if retract_progress < 0.5:
                    # Hold extended position
                    foot[0] = base_pos[0] * self.max_radial_extension
                    foot[1] = base_pos[1] * self.max_radial_extension
                    foot[2] = base_pos[2] + 0.05
                else:
                    # Begin retraction
                    ret_curve = (retract_progress - 0.5) / 0.5
                    radial_mult = self.max_radial_extension - (self.max_radial_extension - 1.0) * ret_curve
                    foot[0] = base_pos[0] * radial_mult
                    foot[1] = base_pos[1] * radial_mult
                    foot[2] = base_pos[2] + 0.05 * (1.0 - ret_curve)
                    
        else:
            # Landing phase [0.85-1.0]: retract to nominal landing position
            progress = (phase - 0.85) / 0.15
            land_curve = np.sin(progress * np.pi / 2)
            
            # Interpolate from current extended position back to nominal
            current_radial = 1.0 + (self.max_radial_extension - 1.0) * (1.0 - land_curve)
            foot[0] = base_pos[0] * current_radial
            foot[1] = base_pos[1] * current_radial
            foot[2] = base_pos[2] * (1.0 + 0.1 * (1.0 - land_curve))  # return to nominal z
        
        return foot