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
        self.max_radial_extension_x = 1.16
        self.max_radial_extension_y = 1.10
        
        # Leg-specific extension limits (rear legs more conservative)
        self.leg_extension_factors = {
            leg_names[0]: 1.0,    # FL - full extension
            leg_names[1]: 1.0,    # FR - full extension
            leg_names[2]: 0.90,   # RR - reduced to protect hip
            leg_names[3]: 0.90,   # RL - reduced to protect hip
        }
        
        # Leg-specific tuck limits (rear legs slightly less to avoid joint stress)
        self.leg_tuck_factors = {
            leg_names[0]: 1.0,    # FL - full tuck
            leg_names[1]: 1.0,    # FR - full tuck
            leg_names[2]: 0.90,   # RR - reduced tuck
            leg_names[3]: 0.90,   # RL - reduced tuck
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
        Compute foot position in body frame with aggressive foot tucking
        during descent to maintain ground clearance.
        
        Key improvements:
        - Increased tuck magnitude: +0.07m maximum (was +0.04m)
        - Extended tuck duration: held until phase 0.88 (was 0.85)
        - Earlier radial retraction: begins at phase 0.60 (was 0.65)
        - Delayed landing descent: begins at phase 0.88 (was 0.85)
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Get leg-specific factors
        leg_factor = self.leg_extension_factors[leg_name]
        tuck_factor = self.leg_tuck_factors[leg_name]
        spiral_offset = self.spiral_phase_offsets[leg_name]
        
        if phase < 0.2:
            # Compression: retract upward and inward
            progress = phase / 0.2
            compression_factor = np.sin(progress * np.pi / 2)
            foot[0] *= (1.0 - 0.24 * compression_factor)
            foot[1] *= (1.0 - 0.24 * compression_factor)
            foot[2] += self.compression_depth * compression_factor
            
        elif phase < 0.4:
            # Launch: explosive downward extension
            progress = (phase - 0.2) / 0.2
            extension_factor = progress**1.3
            foot[0] = base_pos[0] * (0.76 + 0.34 * extension_factor)
            foot[1] = base_pos[1] * (0.76 + 0.34 * extension_factor)
            foot[2] = base_pos[2] + self.compression_depth * (1.0 - extension_factor) - 0.03 * extension_factor
            
        elif phase < 0.6:
            # Early aerial spiral phase: sequential radial extension
            aerial_phase = phase - 0.4
            leg_extension_start = spiral_offset
            leg_extension_duration = 0.18
            
            if aerial_phase < leg_extension_start:
                # Before this leg's extension starts
                foot[0] = base_pos[0] * 1.08
                foot[1] = base_pos[1] * 1.08
                foot[2] = base_pos[2] + 0.01
                
            elif aerial_phase < leg_extension_start + leg_extension_duration:
                # During extension: asymmetric radial outward motion
                ext_progress = (aerial_phase - leg_extension_start) / leg_extension_duration
                ext_curve = ext_progress**1.4
                
                radial_mult_x = 1.08 + (self.max_radial_extension_x - 1.08) * ext_curve * leg_factor
                radial_mult_y = 1.08 + (self.max_radial_extension_y - 1.08) * ext_curve * leg_factor
                
                foot[0] = base_pos[0] * radial_mult_x
                foot[1] = base_pos[1] * radial_mult_y
                foot[2] = base_pos[2] + 0.01
                
            else:
                # Peak extension reached, hold position
                foot[0] = base_pos[0] * (1.08 + (self.max_radial_extension_x - 1.08) * leg_factor)
                foot[1] = base_pos[1] * (1.08 + (self.max_radial_extension_y - 1.08) * leg_factor)
                foot[2] = base_pos[2] + 0.01
                
        elif phase < 0.65:
            # Begin descent preparation: initiate tucking and radial retraction
            progress = (phase - 0.6) / 0.05
            tuck_curve = np.sin(progress * np.pi / 2)
            retract_curve = progress * 0.15  # begin gentle retraction
            
            # Start radial retraction
            current_radial_x = (1.08 + (self.max_radial_extension_x - 1.08) * leg_factor) * (1.0 - retract_curve) + 1.0 * retract_curve
            current_radial_y = (1.08 + (self.max_radial_extension_y - 1.08) * leg_factor) * (1.0 - retract_curve) + 1.0 * retract_curve
            
            foot[0] = base_pos[0] * current_radial_x
            foot[1] = base_pos[1] * current_radial_y
            # Increase tuck from +0.01 to +0.05
            foot[2] = base_pos[2] + (0.01 + 0.04 * tuck_curve) * tuck_factor
            
        elif phase < 0.75:
            # Active descent: maintain moderate tuck, continue retraction
            progress = (phase - 0.65) / 0.10
            retract_curve = 0.15 + 0.35 * np.sin(progress * np.pi / 2)  # retract from 15% to 50%
            
            current_radial_x = (1.08 + (self.max_radial_extension_x - 1.08) * leg_factor) * (1.0 - retract_curve) + 1.0 * retract_curve
            current_radial_y = (1.08 + (self.max_radial_extension_y - 1.08) * leg_factor) * (1.0 - retract_curve) + 1.0 * retract_curve
            
            foot[0] = base_pos[0] * current_radial_x
            foot[1] = base_pos[1] * current_radial_y
            # Maintain +0.05m tuck
            foot[2] = base_pos[2] + 0.05 * tuck_factor
            
        elif phase < 0.88:
            # Deep descent: maximum tuck, complete retraction
            progress = (phase - 0.75) / 0.13
            retract_curve = 0.50 + 0.50 * np.sin(progress * np.pi / 2)  # complete retraction to 100%
            tuck_increase = np.sin(progress * np.pi / 2)
            
            current_radial_x = (1.08 + (self.max_radial_extension_x - 1.08) * leg_factor) * (1.0 - retract_curve) + 1.0 * retract_curve
            current_radial_y = (1.08 + (self.max_radial_extension_y - 1.08) * leg_factor) * (1.0 - retract_curve) + 1.0 * retract_curve
            
            foot[0] = base_pos[0] * current_radial_x
            foot[1] = base_pos[1] * current_radial_y
            # Increase tuck from +0.05 to +0.07 (maximum)
            foot[2] = base_pos[2] + (0.05 + 0.02 * tuck_increase) * tuck_factor
            
        elif phase < 0.97:
            # Landing preparation: smooth descent from maximum tuck to ground contact
            progress = (phase - 0.88) / 0.09
            landing_curve = np.sin(progress * np.pi / 2)
            
            # Feet at nominal radial position
            foot[0] = base_pos[0]
            foot[1] = base_pos[1]
            # Smooth descent from +0.07 to nominal
            foot[2] = base_pos[2] + 0.07 * tuck_factor * (1.0 - landing_curve)
            
        else:
            # Final landing: feet at nominal position for ground contact
            foot[0] = base_pos[0]
            foot[1] = base_pos[1]
            foot[2] = base_pos[2]
        
        return foot