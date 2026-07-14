from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_METRONOME_SWAY_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Metronome sway advance gait with large-amplitude lateral base rolling.
    
    Motion characteristics:
    - All four feet maintain ground contact throughout the cycle
    - Base rolls laterally with smooth sinusoidal pattern
    - Forward surges occur during neutral roll transitions
    - Legs extend/compress with pattern INVERTED from roll to reduce joint stress
    - Conservative parameters to respect joint limits
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5
        
        # Base foot positions (BODY frame) - widen stance for stability
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            foot = v.copy()
            foot[1] *= 1.10  # Widen stance by 10% to reduce roll stress
            self.base_feet_pos_body[k] = foot
        
        # Motion parameters - conservative to avoid joint violations
        self.max_roll_angle = np.deg2rad(10.0)  # Further reduced to 10°
        self.leg_compression_range = 0.10  # Reduced to 0.10m (±0.05m per leg)
        
        # Forward surge parameters
        self.surge_velocity = 0.6
        self.drift_velocity = 0.08
        
        # Lateral motion parameters
        self.lateral_velocity_amp = 0.20
        
        # Vertical motion parameters - minimal
        self.vertical_velocity_amp = 0.01
        
        # Base state - start with elevated base for joint margin
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, 0.04])  # Raise base 4cm
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with lateral rolling and forward surges.
        """
        
        # Roll angle: peak right (+10°) at phase 0.25, peak left (-10°) at phase 0.75
        target_roll = self.max_roll_angle * np.sin(2 * np.pi * phase)
        
        # Roll rate
        roll_rate = self.max_roll_angle * 2 * np.pi * self.freq * np.cos(2 * np.pi * phase)
        
        # Forward velocity: surge during neutral transitions
        if 0.25 <= phase < 0.5 or 0.75 <= phase < 1.0:
            vx = self.surge_velocity
        else:
            vx = self.drift_velocity
        
        # Lateral velocity
        vy = self.lateral_velocity_amp * np.cos(2 * np.pi * phase)
        
        # Minimal vertical oscillation
        vz = self.vertical_velocity_amp * np.sin(4 * np.pi * phase)
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
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
        Compute foot position in body frame with INVERTED compression pattern.
        
        KEY CHANGE: Legs EXTEND when base rolls toward them (to accommodate geometry)
                    Legs COMPRESS when base is neutral (minimal geometric stress)
        
        Right legs (FR, RR):
        - EXTEND during right sway [0.0, 0.25] (base rolling right)
        - Compress during neutral transition [0.25, 0.5]
        - COMPRESS during left sway [0.5, 0.75] (base rolling left)
        - Extend during neutral transition [0.75, 1.0]
        
        Left legs (FL, RL):
        - COMPRESS during right sway [0.0, 0.25] (base rolling right)
        - Extend during neutral transition [0.25, 0.5]
        - EXTEND during left sway [0.5, 0.75] (base rolling left)
        - Compress during neutral transition [0.75, 1.0]
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg side
        is_right_leg = leg_name in ['FR', 'RR']
        
        # INVERTED pattern: legs extend when roll is toward them
        # This is opposite of all previous iterations
        if is_right_leg:
            # Right legs: EXTEND at phase 0.25 (right roll), COMPRESS at phase 0.75 (left roll)
            # Use +sin instead of -sin to invert the pattern
            vertical_offset = self.leg_compression_range * 0.5 * np.sin(2 * np.pi * phase)
        else:
            # Left legs: COMPRESS at phase 0.25 (right roll), EXTEND at phase 0.75 (left roll)
            # Use -sin instead of +sin to invert the pattern
            vertical_offset = -self.leg_compression_range * 0.5 * np.sin(2 * np.pi * phase)
        
        # Apply vertical offset
        foot[2] += vertical_offset
        
        # No lateral adjustment - let base motion handle all lateral displacement
        
        return foot