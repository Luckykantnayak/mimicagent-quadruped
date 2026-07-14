from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_TRIPOD_TILT_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Tripod gait with exaggerated lateral roll tilts.
    
    - Alternates between left-tripod (FL-RR-RL) and right-tripod (FR-RL-RR)
    - Base tilts heavily left during left-tripod stance, right during right-tripod
    - High swing clearance for non-supporting front leg
    - Continuous forward velocity throughout cycle
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for exaggerated tilting motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.step_length = 0.15  # Forward stride per step
        self.step_height = 0.15  # High clearance for exaggerated motion
        self.vx_forward = 0.4  # Constant forward velocity
        
        # Roll tilt parameters
        self.max_roll_angle = np.radians(20)  # Maximum roll tilt (degrees)
        self.roll_rate_mag = 1.5  # Roll rate magnitude (rad/s)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track accumulated roll for smooth transitions
        self.current_roll = 0.0
        
    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward velocity and phase-synchronized roll rate.
        
        Phase structure:
        [0.0, 0.2]: Left tripod stance, tilt left (negative roll rate)
        [0.2, 0.4]: Transition to right, reverse tilt (positive roll rate)
        [0.4, 0.6]: Right tripod stance, tilt right (positive roll rate)
        [0.6, 0.8]: Transition to left, reverse tilt (negative roll rate)
        [0.8, 1.0]: Leveling phase (zero roll rate)
        """
        
        # Constant forward velocity
        vx = self.vx_forward
        
        # Phase-dependent roll rate
        if phase < 0.2:
            # Left tripod stance - tilt left
            roll_rate = -self.roll_rate_mag
        elif phase < 0.4:
            # Transition to right - reverse tilt with smooth ramp
            progress = (phase - 0.2) / 0.2
            roll_rate = self.roll_rate_mag * np.sin(np.pi * progress)
        elif phase < 0.6:
            # Right tripod stance - tilt right
            roll_rate = self.roll_rate_mag
        elif phase < 0.8:
            # Transition to left - reverse tilt with smooth ramp
            progress = (phase - 0.6) / 0.2
            roll_rate = -self.roll_rate_mag * np.sin(np.pi * progress)
        else:
            # Leveling phase - zero rate
            roll_rate = 0.0
        
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
        
        # Track current roll for foot placement compensation
        roll, _, _ = quat_to_euler(self.root_quat)
        self.current_roll = roll
        
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with tripod gait pattern.
        
        FL: stance [0.0-0.4, 0.8-1.0], swing [0.4-0.8]
        FR: swing [0.0-0.4], stance [0.4-1.0]
        RL: stance throughout (participates in both tripods)
        RR: stance throughout (participates in both tripods)
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg-specific phase and motion
        if leg_name.startswith('FL'):
            # FL: Left tripod support, swings during right tripod
            if phase < 0.4:
                # Stance phase - foot planted forward
                foot[0] += self.step_length * 0.5
            elif phase < 0.8:
                # Swing phase - arc forward with high clearance
                swing_progress = (phase - 0.4) / 0.4
                foot[0] += self.step_length * (swing_progress - 0.5)
                foot[2] += self.step_height * np.sin(np.pi * swing_progress)
            else:
                # Landing and re-establishing stance
                foot[0] += self.step_length * 0.5
                
        elif leg_name.startswith('FR'):
            # FR: Right tripod support, swings during left tripod
            if phase < 0.2:
                # Swing phase - arc forward with high clearance
                swing_progress = (phase + 0.2) / 0.4  # Offset to align with left tripod
                if swing_progress > 1.0:
                    swing_progress -= 1.0
                foot[0] += self.step_length * (swing_progress - 0.5)
                foot[2] += self.step_height * np.sin(np.pi * swing_progress)
            elif phase < 0.4:
                # Landing phase
                swing_progress = (phase + 0.2) / 0.4
                foot[0] += self.step_length * (swing_progress - 0.5)
                foot[2] += self.step_height * np.sin(np.pi * swing_progress)
            else:
                # Stance phase - foot planted forward
                foot[0] += self.step_length * 0.5
                
        elif leg_name.startswith('RL'):
            # RL: Rear left - participates in both tripods, mostly stance
            # Slight fore-aft positioning to maintain stable geometry
            if phase < 0.5:
                foot[0] -= self.step_length * 0.3
            else:
                foot[0] -= self.step_length * 0.2
                
        elif leg_name.startswith('RR'):
            # RR: Rear right - participates in both tripods, mostly stance
            # Slight fore-aft positioning to maintain stable geometry
            if phase < 0.5:
                foot[0] -= self.step_length * 0.2
            else:
                foot[0] -= self.step_length * 0.3
        
        # Compensate lateral position for body roll to maintain ground contact
        # When body tilts left (negative roll), right legs need more lateral extension
        # When body tilts right (positive roll), left legs need more lateral extension
        roll_compensation = 0.08  # Lateral adjustment per radian of roll
        
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            # Left side legs - extend more when body tilts right
            foot[1] -= roll_compensation * self.current_roll
        else:
            # Right side legs - extend more when body tilts left
            foot[1] += roll_compensation * self.current_roll
        
        return foot