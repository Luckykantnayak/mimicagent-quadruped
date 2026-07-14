from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PHOENIX_RISE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Phoenix Rise Spin: Dramatic theatrical maneuver where robot rises from compressed
    crouch to maximum height while executing full 360-degree yaw rotation.
    
    All four legs extend LATERALLY outward in synchronized 'wings-spreading' pattern
    while maintaining ground contact throughout.
    
    Phase breakdown:
      [0.0, 0.15]: Compressed crouch, stationary
      [0.15, 0.45]: Rise and extend legs laterally, initiate spin
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
        
        # Motion parameters - conservative for safety
        self.max_rise_height = 0.12  # Reduced further for joint safety
        self.total_rotation = 2 * np.pi  # 360 degrees
        
        # Phase timing
        self.rise_start = 0.15
        self.rise_end = 0.45
        self.hold_start = 0.45
        self.hold_end = 0.55
        self.descend_start = 0.55
        self.descend_end = 0.85
        
        # Vertical velocity tuning
        self.rise_duration = self.rise_end - self.rise_start
        self.descend_duration = self.descend_end - self.descend_start
        self.vz_magnitude = self.max_rise_height / (self.rise_duration / self.freq)
        
        # Yaw rate tuning
        self.spin_duration = 1.0 - self.rise_start
        self.yaw_rate_magnitude = self.total_rotation / (self.spin_duration / self.freq)
        
        # Extension parameters - emphasize LATERAL spreading, minimal longitudinal
        self.compression_factor = 0.80  # Keep joints in safer mid-range
        self.max_lateral_extension_factor = 1.35  # Lateral (y) extension
        self.max_longitudinal_extension_factor = 1.05  # Minimal forward/back (x) extension
        
    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent vertical velocity and yaw rate.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase [0.0, 0.15]: Stationary compressed crouch
        if phase < self.rise_start:
            pass
        
        # Phase [0.15, 0.45]: Rising with upward velocity
        elif self.rise_start <= phase < self.rise_end:
            local_phase = (phase - self.rise_start) / (self.rise_end - self.rise_start)
            # Smooth blending at boundaries
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
        
        # Phase [0.55, 0.85]: Descending
        elif self.descend_start <= phase < self.descend_end:
            local_phase = (phase - self.descend_start) / (self.descend_end - self.descend_start)
            if local_phase < 0.1:
                blend = local_phase / 0.1
            elif local_phase > 0.9:
                blend = (1.0 - local_phase) / 0.1
            else:
                blend = 1.0
            vz = -self.vz_magnitude * blend
            yaw_rate = self.yaw_rate_magnitude
        
        # Phase [0.85, 1.0]: Stationary, complete rotation
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
        Compute foot position in body frame with LATERAL extension pattern.
        
        Legs extend primarily in Y-direction (sideways) to create 'wings-spreading'
        effect while maintaining stable joint configurations. Minimal X-direction
        (forward/backward) change to avoid knee over-extension.
        
        Foot z remains constant in body frame - no dynamic adjustment.
        """
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute extension factors based on phase
        if phase < self.rise_start:
            # Compressed position
            lateral_factor = self.compression_factor
            longitudinal_factor = self.compression_factor
        elif self.rise_start <= phase < self.rise_end:
            # Rising and extending
            progress = (phase - self.rise_start) / (self.rise_end - self.rise_start)
            # Smooth interpolation
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            lateral_factor = self.compression_factor + (self.max_lateral_extension_factor - self.compression_factor) * smooth_progress
            longitudinal_factor = self.compression_factor + (self.max_longitudinal_extension_factor - self.compression_factor) * smooth_progress
        elif self.hold_start <= phase < self.hold_end:
            # Hold maximum extension
            lateral_factor = self.max_lateral_extension_factor
            longitudinal_factor = self.max_longitudinal_extension_factor
        elif self.descend_start <= phase < self.descend_end:
            # Descending and retracting
            progress = (phase - self.descend_start) / (self.descend_end - self.descend_start)
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            lateral_factor = self.max_lateral_extension_factor - (self.max_lateral_extension_factor - self.compression_factor) * smooth_progress
            longitudinal_factor = self.max_longitudinal_extension_factor - (self.max_longitudinal_extension_factor - self.compression_factor) * smooth_progress
        else:
            # Return to compressed position
            lateral_factor = self.compression_factor
            longitudinal_factor = self.compression_factor
        
        # Apply asymmetric extension: emphasize lateral (Y), minimal longitudinal (X)
        foot = base_foot.copy()
        foot[0] = base_foot[0] * longitudinal_factor  # Minimal forward/back change
        foot[1] = base_foot[1] * lateral_factor  # Significant sideways extension
        foot[2] = base_foot[2]  # Constant z in body frame
        
        return foot