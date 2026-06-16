from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_METRONOME_SWAY_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Metronome sway advance gait with large-amplitude lateral base rolling (±30°).
    
    Motion characteristics:
    - Base rolls rhythmically left and right like an inverted pendulum metronome
    - Forward surges occur during neutral roll transitions
    - All four feet maintain continuous ground contact
    - Legs compress/extend asymmetrically to accommodate lateral sway
    - Left and right leg groups exhibit anti-phase vertical motion
    
    Phase structure:
    - [0.0, 0.25]: Right sway - base rolls right (~30°), right legs compress
    - [0.25, 0.5]: Neutral surge 1 - base returns to neutral, forward velocity surge
    - [0.5, 0.75]: Left sway - base rolls left (~-30°), left legs compress
    - [0.75, 1.0]: Neutral surge 2 - base returns to neutral, second forward surge
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slow phase rate for quasi-static stability
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.max_roll_angle = np.deg2rad(30)  # ±30° roll amplitude
        self.surge_velocity = 0.8  # Forward velocity during surge phases
        self.drift_velocity = 0.1  # Minimal forward velocity during sway phases
        self.lateral_shift_velocity = 0.3  # Lateral velocity during sway
        
        # Leg compression/extension parameters
        self.compression_amount = 0.15  # Vertical compression when weight-bearing
        self.lateral_shift_amount = 0.08  # Lateral shift in body frame during sway
        self.rearward_slide = 0.12  # Rearward foot motion during surge phases
        
        # Base state (WORLD frame)
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with lateral rolling and forward surges.
        
        Roll profile:
        - Phase 0.0-0.25: Roll right (0° → +30°)
        - Phase 0.25-0.5: Roll back to neutral (+30° → 0°)
        - Phase 0.5-0.75: Roll left (0° → -30°)
        - Phase 0.75-1.0: Roll back to neutral (-30° → 0°)
        
        Forward velocity:
        - High during neutral transitions (0.25-0.5, 0.75-1.0)
        - Low during sway phases (0.0-0.25, 0.5-0.75)
        """
        
        # Determine sub-phase
        if phase < 0.25:
            # Right sway phase
            sub_phase = phase / 0.25
            roll_rate = self._smooth_rate(sub_phase) * self.max_roll_angle / (0.25 / self.freq)
            vx = self.drift_velocity
            vy = self.lateral_shift_velocity
            vz = -0.05
            
        elif phase < 0.5:
            # Neutral surge 1
            sub_phase = (phase - 0.25) / 0.25
            roll_rate = -self._smooth_rate(sub_phase) * self.max_roll_angle / (0.25 / self.freq)
            vx = self.surge_velocity
            vy = -self.lateral_shift_velocity * (1.0 - sub_phase)
            vz = 0.05
            
        elif phase < 0.75:
            # Left sway phase
            sub_phase = (phase - 0.5) / 0.25
            roll_rate = -self._smooth_rate(sub_phase) * self.max_roll_angle / (0.25 / self.freq)
            vx = self.drift_velocity
            vy = -self.lateral_shift_velocity
            vz = -0.05
            
        else:
            # Neutral surge 2
            sub_phase = (phase - 0.75) / 0.25
            roll_rate = self._smooth_rate(sub_phase) * self.max_roll_angle / (0.25 / self.freq)
            vx = self.surge_velocity
            vy = self.lateral_shift_velocity * (1.0 - sub_phase)
            vz = 0.05
        
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

    def _smooth_rate(self, sub_phase):
        """
        Smooth rate profile using sinusoidal shaping.
        Returns value in [0, 1] with smooth acceleration/deceleration.
        """
        return np.sin(np.pi * sub_phase)

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in BODY frame with compression/extension and lateral shifts.
        
        Left legs (FL, RL):
        - Extend during right sway (0.0-0.25)
        - Compress during left sway (0.5-0.75)
        
        Right legs (FR, RR):
        - Compress during right sway (0.0-0.25)
        - Extend during left sway (0.5-0.75)
        
        All legs:
        - Slide rearward during surge phases (0.25-0.5, 0.75-1.0)
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if left or right leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Phase-dependent foot motion
        if phase < 0.25:
            # Right sway phase
            sub_phase = phase / 0.25
            sway_progress = self._smooth_rate(sub_phase)
            
            if is_right_leg:
                # Right legs compress and shift inward
                foot[2] += self.compression_amount * sway_progress
                foot[1] -= self.lateral_shift_amount * sway_progress
            else:
                # Left legs extend and shift outward
                foot[2] -= self.compression_amount * sway_progress
                foot[1] -= self.lateral_shift_amount * sway_progress
                
        elif phase < 0.5:
            # Neutral surge 1
            sub_phase = (phase - 0.25) / 0.25
            transition_progress = self._smooth_rate(sub_phase)
            
            if is_right_leg:
                # Right legs return to neutral
                foot[2] += self.compression_amount * (1.0 - transition_progress)
                foot[1] -= self.lateral_shift_amount * (1.0 - transition_progress)
            else:
                # Left legs return to neutral
                foot[2] -= self.compression_amount * (1.0 - transition_progress)
                foot[1] -= self.lateral_shift_amount * (1.0 - transition_progress)
            
            # All legs slide rearward during forward surge
            foot[0] -= self.rearward_slide * transition_progress
            
        elif phase < 0.75:
            # Left sway phase
            sub_phase = (phase - 0.5) / 0.25
            sway_progress = self._smooth_rate(sub_phase)
            
            if is_left_leg:
                # Left legs compress and shift inward
                foot[2] += self.compression_amount * sway_progress
                foot[1] += self.lateral_shift_amount * sway_progress
            else:
                # Right legs extend and shift outward
                foot[2] -= self.compression_amount * sway_progress
                foot[1] += self.lateral_shift_amount * sway_progress
                
        else:
            # Neutral surge 2
            sub_phase = (phase - 0.75) / 0.25
            transition_progress = self._smooth_rate(sub_phase)
            
            if is_left_leg:
                # Left legs return to neutral
                foot[2] += self.compression_amount * (1.0 - transition_progress)
                foot[1] += self.lateral_shift_amount * (1.0 - transition_progress)
            else:
                # Right legs return to neutral
                foot[2] -= self.compression_amount * (1.0 - transition_progress)
                foot[1] += self.lateral_shift_amount * (1.0 - transition_progress)
            
            # All legs slide rearward during forward surge
            foot[0] -= self.rearward_slide * (0.5 + transition_progress * 0.5)
        
        return foot