from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_METRONOME_SWAY_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Metronome sway advance gait with large-amplitude lateral base rolling (±30°).
    
    Motion characteristics:
    - All four feet remain in continuous ground contact
    - Base alternates between right-leaning and left-leaning phases
    - Forward surges occur during neutral roll transitions
    - Legs compress/extend asymmetrically to accommodate lateral sway
    - Right legs compress during rightward roll, left legs compress during leftward roll
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for large amplitude swaying motion
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.max_roll_angle = np.deg2rad(30.0)  # ±30° peak roll
        self.forward_surge_velocity = 0.8  # Forward velocity during surge phases
        self.forward_drift_velocity = 0.1  # Minimal forward velocity during sway phases
        self.lateral_velocity_amplitude = 0.3  # Lateral shift velocity
        
        # Leg motion parameters
        self.compression_amplitude = 0.06  # Vertical compression/extension amplitude
        self.lateral_shift_amplitude = 0.03  # Lateral foot shift amplitude
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with lateral rolling and forward surges.
        
        Phase structure:
        [0.0, 0.25]: Right sway - roll right, minimal forward, rightward lateral
        [0.25, 0.5]: Right-to-neutral surge - unwind roll, forward surge, center laterally
        [0.5, 0.75]: Left sway - roll left, minimal forward, leftward lateral
        [0.75, 1.0]: Left-to-neutral surge - unwind roll, forward surge, center laterally
        """
        
        # Compute roll rate to achieve ±30° peaks at phases 0.25 and 0.75
        # Roll angle profile: 0° -> +30° -> 0° -> -30° -> 0°
        if phase < 0.25:
            # Rolling right: 0° to +30°
            roll_rate = 4.0 * self.max_roll_angle * self.freq
            vx = self.forward_drift_velocity
            vy = self.lateral_velocity_amplitude
        elif phase < 0.5:
            # Unwinding right roll: +30° to 0°
            roll_rate = -4.0 * self.max_roll_angle * self.freq
            vx = self.forward_surge_velocity
            # Smooth transition from rightward to centered
            sub_phase = (phase - 0.25) / 0.25
            vy = self.lateral_velocity_amplitude * (1.0 - 2.0 * sub_phase)
        elif phase < 0.75:
            # Rolling left: 0° to -30°
            roll_rate = -4.0 * self.max_roll_angle * self.freq
            vx = self.forward_drift_velocity
            vy = -self.lateral_velocity_amplitude
        else:
            # Unwinding left roll: -30° to 0°
            roll_rate = 4.0 * self.max_roll_angle * self.freq
            vx = self.forward_surge_velocity
            # Smooth transition from leftward to centered
            sub_phase = (phase - 0.75) / 0.25
            vy = -self.lateral_velocity_amplitude * (1.0 - 2.0 * sub_phase)
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, 0.0])
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
        Compute foot position in body frame with compression/extension and lateral shifts.
        
        Right legs (FR, RR): compress during [0.0, 0.25], extend during [0.5, 0.75]
        Left legs (FL, RL): extend during [0.0, 0.25], compress during [0.5, 0.75]
        All legs return to neutral during transition phases [0.25, 0.5] and [0.75, 1.0]
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a right-side or left-side leg
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compute vertical compression/extension and lateral shift
        if phase < 0.25:
            # Right sway phase
            progress = phase / 0.25
            if is_right_leg:
                # Right legs compress (move up in body frame)
                foot[2] += self.compression_amplitude * progress
                # Slight inward lateral shift
                foot[1] -= self.lateral_shift_amplitude * progress
            else:  # Left legs
                # Left legs extend (move down in body frame)
                foot[2] -= self.compression_amplitude * progress
                # Slight outward lateral shift
                foot[1] -= self.lateral_shift_amplitude * progress
                
        elif phase < 0.5:
            # Right-to-neutral transition
            progress = (phase - 0.25) / 0.25
            if is_right_leg:
                # Right legs extend back to neutral
                foot[2] += self.compression_amplitude * (1.0 - progress)
                foot[1] -= self.lateral_shift_amplitude * (1.0 - progress)
            else:  # Left legs
                # Left legs retract back to neutral
                foot[2] -= self.compression_amplitude * (1.0 - progress)
                foot[1] -= self.lateral_shift_amplitude * (1.0 - progress)
                
        elif phase < 0.75:
            # Left sway phase
            progress = (phase - 0.5) / 0.25
            if is_left_leg:
                # Left legs compress (move up in body frame)
                foot[2] += self.compression_amplitude * progress
                # Slight inward lateral shift
                foot[1] += self.lateral_shift_amplitude * progress
            else:  # Right legs
                # Right legs extend (move down in body frame)
                foot[2] -= self.compression_amplitude * progress
                # Slight outward lateral shift
                foot[1] += self.lateral_shift_amplitude * progress
                
        else:
            # Left-to-neutral transition
            progress = (phase - 0.75) / 0.25
            if is_left_leg:
                # Left legs extend back to neutral
                foot[2] += self.compression_amplitude * (1.0 - progress)
                foot[1] += self.lateral_shift_amplitude * (1.0 - progress)
            else:  # Right legs
                # Right legs retract back to neutral
                foot[2] -= self.compression_amplitude * (1.0 - progress)
                foot[1] += self.lateral_shift_amplitude * (1.0 - progress)
        
        return foot