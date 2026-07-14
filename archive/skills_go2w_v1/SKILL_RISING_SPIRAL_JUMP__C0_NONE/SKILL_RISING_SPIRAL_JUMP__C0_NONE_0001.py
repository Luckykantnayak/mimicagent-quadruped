from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising spiral jump: vertical jump with yaw rotation and sequential spiral leg extension.
    
    Phase breakdown:
      [0.0, 0.2]: Compression - all legs compress, base lowers
      [0.2, 0.4]: Launch - explosive extension, upward velocity, yaw rotation starts
      [0.4, 0.6]: Aerial spiral expansion - legs extend sequentially (FL→FR→RR→RL)
      [0.6, 0.8]: Peak spiral - maximum height, legs fully extended
      [0.8, 1.0]: Descent and landing - legs retract, base descends, contact re-established
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower cycle for jump timing
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.compression_height = 0.08  # How much legs compress (foot moves up in body frame)
        self.spiral_extension_radius = 0.15  # How far legs extend outward during spiral
        self.launch_velocity = 1.5  # Peak upward velocity during launch
        self.yaw_rate_max = 3.0  # Maximum yaw angular velocity (rad/s)
        self.gravity = 2.0  # Effective gravity for descent shaping
        
        # Spiral timing offsets within phase [0.4, 0.6]
        # FL starts earliest, RL starts latest
        self.spiral_phase_offsets = {
            'FL': 0.0,
            'FR': 0.333,
            'RR': 0.667,
            'RL': 1.0
        }
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase [0.0, 0.2]: Compression - slight downward motion
        if phase < 0.2:
            local_phase = phase / 0.2
            vz = -0.3 * (1.0 - local_phase)  # Slight downward velocity, goes to zero
        
        # Phase [0.2, 0.4]: Launch - explosive upward velocity and yaw initiation
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Ramp up upward velocity
            vz = self.launch_velocity * np.sin(np.pi * local_phase * 0.5)
            # Ramp up yaw rate
            yaw_rate = self.yaw_rate_max * local_phase
        
        # Phase [0.4, 0.6]: Aerial spiral expansion - upward velocity decreases
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Decelerate upward velocity (gravity effect)
            vz = self.launch_velocity * (1.0 - local_phase)
            # Constant yaw rate
            yaw_rate = self.yaw_rate_max
        
        # Phase [0.6, 0.8]: Peak spiral - transition to downward
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Transition from zero to downward velocity
            vz = -self.gravity * local_phase
            # Constant yaw rate
            yaw_rate = self.yaw_rate_max
        
        # Phase [0.8, 1.0]: Descent and landing - dampen downward velocity and yaw
        else:
            local_phase = (phase - 0.8) / 0.2
            # Downward velocity dampens to zero for landing
            vz = -self.gravity * (1.0 - local_phase**2)
            # Yaw rate ramps down to zero
            yaw_rate = self.yaw_rate_max * (1.0 - local_phase)
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
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
        
        # Determine leg-specific direction for spiral extension
        if leg_name.startswith('FL'):
            angle_offset = 0.0  # Forward-left
        elif leg_name.startswith('FR'):
            angle_offset = -np.pi / 2  # Forward-right
        elif leg_name.startswith('RR'):
            angle_offset = -np.pi  # Rear-right
        elif leg_name.startswith('RL'):
            angle_offset = np.pi / 2  # Rear-left
        else:
            angle_offset = 0.0
        
        # Phase [0.0, 0.2]: Compression
        if phase < 0.2:
            local_phase = phase / 0.2
            # Foot moves upward in body frame (leg shortens)
            compression = self.compression_height * local_phase
            foot[2] += compression
        
        # Phase [0.2, 0.4]: Launch
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Return from compression and begin liftoff
            compression = self.compression_height * (1.0 - local_phase)
            foot[2] += compression
            # Begin slight outward motion
            extension = 0.05 * local_phase
            foot[0] += extension * np.cos(angle_offset)
            foot[1] += extension * np.sin(angle_offset)
        
        # Phase [0.4, 0.6]: Aerial spiral expansion
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Each leg extends at different timing based on spiral sequence
            spiral_offset = self.spiral_phase_offsets.get(leg_name[:2], 0.0)
            # Compute leg-specific extension progress
            leg_extension_phase = np.clip((local_phase - spiral_offset * 0.3) / (1.0 - spiral_offset * 0.3), 0.0, 1.0)
            # Smooth extension using sine curve
            extension_factor = np.sin(np.pi * leg_extension_phase * 0.5)
            extension = self.spiral_extension_radius * extension_factor
            foot[0] += extension * np.cos(angle_offset)
            foot[1] += extension * np.sin(angle_offset)
            # Slight upward lift during extension
            foot[2] += 0.05 * extension_factor
        
        # Phase [0.6, 0.8]: Peak spiral - maintain extended position
        elif phase < 0.8:
            # Full extension maintained
            extension = self.spiral_extension_radius
            foot[0] += extension * np.cos(angle_offset)
            foot[1] += extension * np.sin(angle_offset)
            foot[2] += 0.05
        
        # Phase [0.8, 1.0]: Descent and landing - retract legs
        else:
            local_phase = (phase - 0.8) / 0.2
            # Retract from extended position back to nominal
            retraction_factor = 1.0 - local_phase
            extension = self.spiral_extension_radius * retraction_factor
            foot[0] += extension * np.cos(angle_offset)
            foot[1] += extension * np.sin(angle_offset)
            foot[2] += 0.05 * retraction_factor
            # Prepare for landing by moving slightly downward at end
            if local_phase > 0.7:
                landing_phase = (local_phase - 0.7) / 0.3
                foot[2] -= 0.03 * landing_phase
        
        return foot