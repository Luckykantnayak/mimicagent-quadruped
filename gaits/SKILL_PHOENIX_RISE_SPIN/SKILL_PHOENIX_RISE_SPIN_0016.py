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
      [0.0, 0.15]: Compressed crouch, stationary
      [0.15, 0.45]: Rise and extend legs, initiate spin
      [0.45, 0.55]: Hold peak height and extension
      [0.55, 0.85]: Descend and retract legs, continue spin
      [0.85, 1.0]: Return to crouch, complete 360-degree rotation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Store base foot positions (nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters - reduced for safety and joint limits
        self.max_rise_height = 0.15  # Conservative rise to stay within height envelope
        self.total_rotation = 2 * np.pi  # 360 degrees
        
        # Phase timing
        self.rise_start = 0.15
        self.rise_end = 0.45
        self.hold_start = 0.45
        self.hold_end = 0.55
        self.descend_start = 0.55
        self.descend_end = 0.85
        
        # Vertical velocity tuning
        self.rise_duration = self.rise_end - self.rise_start  # 0.3 phase units
        self.descend_duration = self.descend_end - self.descend_start  # 0.3 phase units
        self.vz_magnitude = self.max_rise_height / (self.rise_duration / self.freq)
        
        # Yaw rate tuning (active throughout 0.15-1.0)
        self.spin_duration = 1.0 - self.rise_start  # 0.85 phase units
        self.yaw_rate_magnitude = self.total_rotation / (self.spin_duration / self.freq)
        
        # Radial extension parameters - reduced to avoid joint limits
        self.compression_factor = 0.65  # Moderate compression to avoid over-flexion
        self.max_extension_factor = 1.25  # Conservative extension to avoid over-extension
        
    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent vertical velocity and yaw rate.
        
        Vertical velocity profile:
          [0.0, 0.15]: stationary
          [0.15, 0.45]: upward (rise)
          [0.45, 0.55]: stationary (hold peak)
          [0.55, 0.85]: downward (descend)
          [0.85, 1.0]: stationary (return to crouch)
        
        Yaw rate active throughout [0.15, 1.0]
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase [0.0, 0.15]: Stationary compressed crouch
        if phase < self.rise_start:
            pass  # All velocities zero
        
        # Phase [0.15, 0.45]: Rising with upward velocity
        elif self.rise_start <= phase < self.rise_end:
            # Smooth ramp up at start
            local_phase = (phase - self.rise_start) / (self.rise_end - self.rise_start)
            if local_phase < 0.1:
                blend = local_phase / 0.1
            elif local_phase > 0.9:
                blend = (1.0 - local_phase) / 0.1
            else:
                blend = 1.0
            vz = self.vz_magnitude * blend
            yaw_rate = self.yaw_rate_magnitude
        
        # Phase [0.45, 0.55]: Hold peak height
        elif self.hold_start <= phase < self.hold_end:
            vz = 0.0
            yaw_rate = self.yaw_rate_magnitude
        
        # Phase [0.55, 0.85]: Descending with downward velocity
        elif self.descend_start <= phase < self.descend_end:
            # Smooth ramp for descent
            local_phase = (phase - self.descend_start) / (self.descend_end - self.descend_start)
            if local_phase < 0.1:
                blend = local_phase / 0.1
            elif local_phase > 0.9:
                blend = (1.0 - local_phase) / 0.1
            else:
                blend = 1.0
            vz = -self.vz_magnitude * blend
            yaw_rate = self.yaw_rate_magnitude
        
        # Phase [0.85, 1.0]: Stationary at crouch, complete rotation
        else:
            vz = 0.0
            yaw_rate = self.yaw_rate_magnitude
        
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
        
        All legs extend radially outward from compressed position during rise,
        hold at maximum during peak, then retract during descent.
        
        Foot z is adjusted to maintain ground contact as base rises.
        """
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute extension factor based on phase
        if phase < self.rise_start:
            # Compressed position
            extension = self.compression_factor
            height_offset = 0.0
        elif self.rise_start <= phase < self.rise_end:
            # Rising and extending
            progress = (phase - self.rise_start) / (self.rise_end - self.rise_start)
            # Smooth interpolation using cosine
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            extension = self.compression_factor + (self.max_extension_factor - self.compression_factor) * smooth_progress
            height_offset = self.max_rise_height * smooth_progress
        elif self.hold_start <= phase < self.hold_end:
            # Hold maximum extension
            extension = self.max_extension_factor
            height_offset = self.max_rise_height
        elif self.descend_start <= phase < self.descend_end:
            # Descending and retracting
            progress = (phase - self.descend_start) / (self.descend_end - self.descend_start)
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            extension = self.max_extension_factor - (self.max_extension_factor - self.compression_factor) * smooth_progress
            height_offset = self.max_rise_height * (1.0 - smooth_progress)
        else:
            # Return to compressed position
            extension = self.compression_factor
            height_offset = 0.0
        
        # Apply radial extension in x-y plane
        foot = base_foot.copy()
        foot[0] = base_foot[0] * extension
        foot[1] = base_foot[1] * extension
        
        # Adjust z to maintain ground contact as base rises
        # As base rises by height_offset, foot z in body frame must decrease by same amount
        foot[2] = base_foot[2] - height_offset
        
        return foot