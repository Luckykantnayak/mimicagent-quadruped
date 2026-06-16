from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SIDE_SAIL_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Side sail drift motion: cyclic lateral drifting via asymmetric leg extension
    patterns coupled with body roll.
    
    Phase structure:
      [0.0, 0.3]: right_sail - right legs extended, left legs compact, roll right, drift right
      [0.3, 0.5]: transition_to_left - legs reconfigure, roll neutralizes
      [0.5, 0.8]: left_sail - left legs extended, right legs compact, roll left, drift left
      [0.8, 1.0]: transition_to_right - legs reconfigure back, roll returns
    
    All four feet maintain ground contact throughout the entire cycle.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower cycle for smooth drifting motion
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Asymmetric stance parameters
        self.lateral_extension = 0.15  # How far to extend legs laterally (meters)
        self.compact_factor = 0.5  # How much to compact legs inward (0.5 = 50% of base width)
        
        # Base motion parameters
        self.lateral_vel_amplitude = 0.3  # m/s lateral velocity magnitude
        self.roll_rate_amplitude = 0.4  # rad/s roll rate magnitude (~ 23 deg/s)
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity and angular velocity based on phase.
        Creates alternating lateral drift with coordinated roll motion.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        if 0.0 <= phase < 0.3:
            # Right sail phase: drift right, roll right
            progress = phase / 0.3
            # Smooth ramp-up using cosine
            envelope = 0.5 * (1.0 - np.cos(np.pi * progress))
            vy = self.lateral_vel_amplitude * envelope
            roll_rate = self.roll_rate_amplitude * envelope
            
        elif 0.3 <= phase < 0.5:
            # Transition to left: reverse lateral velocity and roll rate
            progress = (phase - 0.3) / 0.2
            # Smooth transition from right to left using cosine interpolation
            transition = np.cos(np.pi * progress)  # +1 to -1
            vy = self.lateral_vel_amplitude * transition
            roll_rate = self.roll_rate_amplitude * transition
            
        elif 0.5 <= phase < 0.8:
            # Left sail phase: drift left, roll left
            progress = (phase - 0.5) / 0.3
            # Smooth ramp-up in opposite direction
            envelope = 0.5 * (1.0 - np.cos(np.pi * progress))
            vy = -self.lateral_vel_amplitude * envelope
            roll_rate = -self.roll_rate_amplitude * envelope
            
        else:  # 0.8 <= phase < 1.0
            # Transition to right: reverse back to prepare for next cycle
            progress = (phase - 0.8) / 0.2
            # Smooth transition from left back to right
            transition = -np.cos(np.pi * progress)  # -1 to +1
            vy = self.lateral_vel_amplitude * transition
            roll_rate = self.roll_rate_amplitude * transition
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase.
        Right legs (FR, RR) extend during right_sail, compact during left_sail.
        Left legs (FL, RL) compact during right_sail, extend during left_sail.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a right or left leg
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        is_front_leg = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Compute extension/compaction factor based on phase
        if 0.0 <= phase < 0.3:
            # Right sail: right legs extended, left legs compact
            progress = phase / 0.3
            blend = 0.5 * (1.0 - np.cos(np.pi * progress))  # Smooth 0->1
            if is_right_leg:
                lateral_offset = self.lateral_extension * blend
            else:
                lateral_offset = -self.lateral_extension * self.compact_factor * blend
                
        elif 0.3 <= phase < 0.5:
            # Transition to left
            progress = (phase - 0.3) / 0.2
            blend = 0.5 * (1.0 + np.cos(np.pi * progress))  # Smooth 1->0
            if is_right_leg:
                # Right leg goes from extended to compact
                lateral_offset = self.lateral_extension * blend - self.lateral_extension * self.compact_factor * (1.0 - blend)
            else:
                # Left leg goes from compact to extended
                lateral_offset = -self.lateral_extension * self.compact_factor * blend + self.lateral_extension * (1.0 - blend)
                
        elif 0.5 <= phase < 0.8:
            # Left sail: left legs extended, right legs compact
            progress = (phase - 0.5) / 0.3
            blend = 0.5 * (1.0 - np.cos(np.pi * progress))  # Smooth 0->1
            if is_right_leg:
                lateral_offset = -self.lateral_extension * self.compact_factor * blend
            else:
                lateral_offset = self.lateral_extension * blend
                
        else:  # 0.8 <= phase < 1.0
            # Transition to right
            progress = (phase - 0.8) / 0.2
            blend = 0.5 * (1.0 + np.cos(np.pi * progress))  # Smooth 1->0
            if is_right_leg:
                # Right leg goes from compact to extended
                lateral_offset = -self.lateral_extension * self.compact_factor * blend + self.lateral_extension * (1.0 - blend)
            else:
                # Left leg goes from extended to compact
                lateral_offset = self.lateral_extension * blend - self.lateral_extension * self.compact_factor * (1.0 - blend)
        
        # Apply lateral offset to y-coordinate (lateral in body frame)
        foot_pos = base_pos.copy()
        foot_pos[1] += lateral_offset
        
        # Keep z-coordinate (height) constant for ground contact
        # x-coordinate remains at base position
        
        return foot_pos