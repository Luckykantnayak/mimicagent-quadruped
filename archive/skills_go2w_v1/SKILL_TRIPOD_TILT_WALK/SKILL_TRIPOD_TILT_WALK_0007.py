from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_TRIPOD_TILT_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Tripod gait with exaggerated lateral roll tilts.
    
    - Alternates between left-tripod (FL-RR-RL) and right-tripod (FR-RL-RR)
    - Base tilts left during left-tripod stance, right during right-tripod
    - High swing clearance for non-supporting front leg
    - Continuous forward velocity throughout cycle
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for exaggerated tilting motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - reduced from first iteration for safety
        self.step_length = 0.10  # Reduced forward stride
        self.step_height = 0.10  # Reduced clearance
        self.vx_forward = 0.35  # Moderate forward velocity
        
        # Roll tilt parameters - reduced magnitude
        self.max_roll_angle = np.radians(12)  # Reduced from 20 to 12 degrees
        self.roll_rate_mag = 1.0  # Moderate roll rate
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track accumulated roll
        self.current_roll = 0.0
        
    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward velocity and phase-synchronized roll rate.
        """
        
        # Constant forward velocity
        vx = self.vx_forward
        
        # Phase-dependent roll rate with smooth transitions
        if phase < 0.2:
            # Left tripod stance - tilt left
            roll_rate = -self.roll_rate_mag
        elif phase < 0.4:
            # Transition to right - smooth sinusoidal ramp
            progress = (phase - 0.2) / 0.2
            roll_rate = self.roll_rate_mag * np.sin(np.pi * progress)
        elif phase < 0.6:
            # Right tripod stance - tilt right
            roll_rate = self.roll_rate_mag
        elif phase < 0.8:
            # Transition to left - smooth sinusoidal ramp
            progress = (phase - 0.6) / 0.2
            roll_rate = -self.roll_rate_mag * np.sin(np.pi * progress)
        else:
            # Leveling phase - smooth decay to zero
            progress = (phase - 0.8) / 0.2
            decay = np.cos(np.pi * progress * 0.5)
            roll_rate = -self.roll_rate_mag * 0.3 * decay
        
        # Set velocity commands
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Track current roll
        roll, _, _ = quat_to_euler(self.root_quat)
        self.current_roll = roll
        
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with tripod gait pattern.
        Includes body-frame z-adjustment to compensate for roll tilt geometry.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Get lateral position for roll compensation
        lateral_offset = foot[1]  # Positive for right legs, negative for left
        
        # Determine leg-specific phase and motion
        if leg_name.startswith('FL'):
            # FL: Left tripod support [0.0-0.4, 0.8-1.0], swing [0.4-0.8]
            if phase < 0.4:
                # Stance phase - foot planted forward
                foot[0] += self.step_length * 0.3
                # Z-compensation: when tilted left (negative roll), left side drops
                # Need to lower left foot in body frame (negative lateral_offset)
                foot[2] -= lateral_offset * np.sin(self.current_roll)
            elif phase < 0.8:
                # Swing phase - arc forward with moderate clearance
                swing_progress = (phase - 0.4) / 0.4
                foot[0] += self.step_length * (swing_progress - 0.2)
                # Reduce swing height during extreme roll
                roll_factor = 1.0 - 0.3 * abs(self.current_roll) / self.max_roll_angle
                effective_height = self.step_height * roll_factor
                foot[2] += effective_height * np.sin(np.pi * swing_progress)
            else:
                # Landing and re-establishing stance
                foot[0] += self.step_length * 0.3
                foot[2] -= lateral_offset * np.sin(self.current_roll)
                
        elif leg_name.startswith('FR'):
            # FR: swing [0.8-1.0, 0.0-0.2], stance [0.2-1.0]
            if phase >= 0.8 or phase < 0.2:
                # Swing phase
                if phase >= 0.8:
                    swing_progress = (phase - 0.8) / 0.4
                else:
                    swing_progress = (phase + 0.2) / 0.4
                foot[0] += self.step_length * (swing_progress - 0.2)
                roll_factor = 1.0 - 0.3 * abs(self.current_roll) / self.max_roll_angle
                effective_height = self.step_height * roll_factor
                foot[2] += effective_height * np.sin(np.pi * swing_progress)
            else:
                # Stance phase
                foot[0] += self.step_length * 0.3
                foot[2] -= lateral_offset * np.sin(self.current_roll)
                
        elif leg_name.startswith('RL'):
            # RL: Participates in both tripods, mostly stance
            if phase < 0.5:
                foot[0] -= self.step_length * 0.2
            else:
                foot[0] -= self.step_length * 0.15
            # Always apply z-compensation during stance
            foot[2] -= lateral_offset * np.sin(self.current_roll)
                
        elif leg_name.startswith('RR'):
            # RR: Participates in both tripods, mostly stance
            if phase < 0.5:
                foot[0] -= self.step_length * 0.15
            else:
                foot[0] -= self.step_length * 0.2
            # Always apply z-compensation during stance
            foot[2] -= lateral_offset * np.sin(self.current_roll)
        
        return foot