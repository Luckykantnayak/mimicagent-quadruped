from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising Spiral Jump: Vertical jump with continuous yaw rotation and sequential
    spiral leg extension during flight.
    
    Phase breakdown:
      [0.0, 0.2]: Compression - all legs compress, base lowers
      [0.2, 0.4]: Launch - explosive upward velocity, yaw begins
      [0.4, 0.6]: Aerial spiral extension - legs extend sequentially FL→FR→RR→RL
      [0.6, 0.8]: Peak hold - full spiral maintained at apex
      [0.8, 1.0]: Descent and landing - legs retract, yaw decelerates, landing
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower cycle for dramatic aerial maneuver
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - REDUCED launch velocity to stay within height envelope
        self.compression_depth = 0.08  # How much base lowers during compression
        self.launch_velocity = 0.65  # Reduced from 2.5 to stay within [0.1, 0.68] height envelope
        self.yaw_rate_max = 3.0  # rad/s, approximately 170 deg/s for visible rotation
        self.spiral_extension_radius = 0.20  # Reduced slightly for stability
        
        # Spiral timing offsets (phase at which each leg begins extension)
        self.spiral_phases = {}
        for leg in leg_names:
            if leg.startswith('FL'):
                self.spiral_phases[leg] = 0.4
            elif leg.startswith('FR'):
                self.spiral_phases[leg] = 0.475
            elif leg.startswith('RR'):
                self.spiral_phases[leg] = 0.5
            elif leg.startswith('RL'):
                self.spiral_phases[leg] = 0.525
        
        self.spiral_extension_duration = 0.075  # Time to fully extend each leg
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.initial_height = 0.0

    def reset(self, root_pos, root_quat):
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.initial_height = root_pos[2]
        self.t = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        Velocity profile tuned to keep base height within [0.1, 0.68] m envelope.
        """
        
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Compression phase: slight downward motion
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth downward then stop
            vz = -0.4 * np.sin(np.pi * local_phase)
        
        # Launch phase: upward velocity with reduced magnitude, yaw starts
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Smooth ramp up to peak launch velocity
            vz = self.launch_velocity * np.sin(np.pi * local_phase)
            yaw_rate = self.yaw_rate_max
        
        # Aerial spiral extension: upward velocity decreasing (gravity effect)
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Steeper deceleration to limit height gain
            vz = self.launch_velocity * (1.0 - local_phase) ** 1.5
            yaw_rate = self.yaw_rate_max
        
        # Peak hold: velocity crosses zero from up to down
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Smooth transition through zero
            vz = self.launch_velocity * 0.3 * np.cos(np.pi * local_phase)
            yaw_rate = self.yaw_rate_max
        
        # Descent and landing: downward velocity decreases, yaw decelerates
        else:
            local_phase = (phase - 0.8) / 0.2
            # Smooth landing approach
            vz = -self.launch_velocity * 0.4 * (1.0 - local_phase) ** 0.5
            # Yaw rate decelerates to zero
            yaw_rate = self.yaw_rate_max * (1.0 - local_phase) ** 2
        
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
          - Compression: move closer to body (z up in body frame)
          - Launch: extend downward
          - Spiral extension: radial outward extension at scheduled phase
          - Peak hold: maintain extended position
          - Landing: retract back to nominal position
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        base_x = foot[0]
        base_y = foot[1]
        base_z = foot[2]
        
        # Determine leg angular position for spiral direction
        angle = np.arctan2(base_y, base_x)
        
        # Compression phase [0.0, 0.2]: legs compress, feet move up relative to base
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth compression using sine
            compression = self.compression_depth * np.sin(np.pi * local_phase)
            foot[2] = base_z + compression
        
        # Launch phase [0.2, 0.4]: legs extend downward for push-off
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Smooth extension downward from compressed state
            compression = self.compression_depth * np.cos(np.pi * 0.5 * local_phase)
            foot[2] = base_z + compression
        
        # Aerial and landing phases [0.4, 1.0]: spiral extension and retraction
        else:
            # Determine spiral extension progress for this leg
            spiral_start = self.spiral_phases[leg_name]
            spiral_end = spiral_start + self.spiral_extension_duration
            
            extension_factor = 0.0
            
            # Extension phase: leg extends radially
            if phase >= spiral_start and phase < spiral_end:
                local_phase = (phase - spiral_start) / self.spiral_extension_duration
                # Smooth extension using sine
                extension_factor = np.sin(np.pi * 0.5 * local_phase)
            
            # Hold extended phase: maintain full extension
            elif phase >= spiral_end and phase < 0.8:
                extension_factor = 1.0
            
            # Retraction phase [0.8, 1.0]: legs retract back
            elif phase >= 0.8:
                local_phase = (phase - 0.8) / 0.2
                # Smooth retraction
                if phase >= spiral_end:
                    extension_factor = 1.0 * np.cos(np.pi * 0.5 * local_phase)
                else:
                    # If leg hadn't fully extended yet, retract from current position
                    if phase >= spiral_start:
                        partial_ext = (phase - spiral_start) / self.spiral_extension_duration
                        extension_factor = np.sin(np.pi * 0.5 * partial_ext) * np.cos(np.pi * 0.5 * local_phase)
                    else:
                        extension_factor = 0.0
            
            # Apply radial extension in x-y plane
            radial_extension = self.spiral_extension_radius * extension_factor
            foot[0] = base_x + radial_extension * np.cos(angle)
            foot[1] = base_y + radial_extension * np.sin(angle)
            
            # During landing phase, smoothly bring z back to nominal
            if phase >= 0.8:
                local_phase = (phase - 0.8) / 0.2
                # Smooth transition back to base z with slight lift for landing prep
                lift = 0.02 * np.sin(np.pi * (1.0 - local_phase))
                foot[2] = base_z + lift
            else:
                # During aerial phase, maintain nominal z in body frame
                foot[2] = base_z
        
        return foot