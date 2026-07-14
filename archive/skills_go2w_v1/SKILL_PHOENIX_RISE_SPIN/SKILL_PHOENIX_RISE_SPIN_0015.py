from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PHOENIX_RISE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Phoenix Rise Spin: Dramatic theatrical maneuver where robot rises from compressed
    crouch to maximum height while executing full 360-degree yaw rotation.
    
    All four legs extend radially outward in synchronized 'wings-spreading' pattern
    while maintaining ground contact throughout.
    
    Phase breakdown:
      [0.0, 0.2]: Compressed crouch, stationary
      [0.2, 0.4]: Initiate rise and spin
      [0.4, 0.6]: Mid-ascent, half rotation (~180 deg)
      [0.6, 0.8]: Approach peak height, continue rotation
      [0.8, 1.0]: Hold peak, complete 360-degree rotation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Store base foot positions (compressed stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.max_rise_height = 0.4  # Total vertical rise distance
        self.total_rotation = 2 * np.pi  # 360 degrees
        
        # Vertical velocity tuning (active phases 0.2-0.8)
        self.rise_duration = 0.6  # Phase duration for rise (0.8 - 0.2)
        self.vz_magnitude = self.max_rise_height / (self.rise_duration / self.freq)
        
        # Yaw rate tuning (active phases 0.2-1.0)
        self.spin_duration = 0.8  # Phase duration for spin (1.0 - 0.2)
        self.yaw_rate_magnitude = self.total_rotation / (self.spin_duration / self.freq)
        
        # Radial extension parameters
        self.compression_factor = 0.3  # At phase 0, feet at 30% of base position radius
        self.max_extension_factor = 1.8  # At phase 1, feet at 180% of base position radius
        
    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent vertical velocity and yaw rate.
        
        Vertical velocity active during [0.2, 0.8]
        Yaw rate active during [0.2, 1.0]
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase [0.0, 0.2]: Stationary compressed crouch
        if phase < 0.2:
            pass  # All velocities remain zero
        
        # Phase [0.2, 0.8]: Rising with constant upward velocity
        elif 0.2 <= phase < 0.8:
            vz = self.vz_magnitude
            yaw_rate = self.yaw_rate_magnitude
        
        # Phase [0.8, 1.0]: Hold peak height, complete rotation
        else:
            vz = 0.0
            yaw_rate = self.yaw_rate_magnitude
        
        # Apply smoothing at phase boundaries to avoid discontinuities
        if 0.18 <= phase < 0.22:
            # Smooth ramp-up at phase 0.2
            blend = (phase - 0.18) / 0.04
            blend = np.clip(blend, 0.0, 1.0)
            vz *= blend
            yaw_rate *= blend
        
        if 0.78 <= phase < 0.82:
            # Smooth ramp-down for vz at phase 0.8
            blend = 1.0 - (phase - 0.78) / 0.04
            blend = np.clip(blend, 0.0, 1.0)
            vz *= blend
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
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
        Compute foot position in body frame with radial extension pattern.
        
        All legs extend radially outward from compressed position (phase 0)
        to maximum extension (phase 1), creating symmetric 'wings-spreading' effect.
        
        Legs maintain ground contact throughout (z = base_z).
        """
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute extension factor: smooth progression from compressed to fully extended
        if phase < 0.2:
            # Compressed position
            extension = self.compression_factor
        elif phase < 0.8:
            # Smoothly extend from compressed to maximum
            progress = (phase - 0.2) / 0.6
            extension = self.compression_factor + (self.max_extension_factor - self.compression_factor) * progress
        else:
            # Hold maximum extension
            extension = self.max_extension_factor
        
        # Apply smooth interpolation using cosine for natural motion
        if 0.2 <= phase < 0.8:
            progress = (phase - 0.2) / 0.6
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            extension = self.compression_factor + (self.max_extension_factor - self.compression_factor) * smooth_progress
        
        # Radial extension: scale x and y from body center
        foot = base_foot.copy()
        foot[0] = base_foot[0] * extension
        foot[1] = base_foot[1] * extension
        foot[2] = base_foot[2]  # Maintain ground contact (z unchanged)
        
        return foot