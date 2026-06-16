from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SCISSORS_KICK_LATERAL_MotionGenerator(BaseMotionGenerator):
    """
    Lateral locomotion skill using alternating vertical scissor kicks.
    
    Right legs (FR up, RR down) kick first to generate rightward impulse,
    then left legs (FL up, RL down) kick to generate leftward impulse.
    
    Phase structure:
    - [0.0, 0.2]: Right scissor kick (FR up, RR down)
    - [0.2, 0.4]: Right legs return to neutral
    - [0.4, 0.6]: Left scissor kick (FL up, RL down)
    - [0.6, 0.8]: Left legs return to neutral
    - [0.8, 1.0]: Settle phase, all legs neutral
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Scissor kick parameters - reduced amplitude to respect joint limits
        self.scissor_amplitude_up = 0.08  # Upward extension for front legs
        self.scissor_amplitude_down = 0.03  # Limited downward extension to avoid ground penetration
        self.lateral_velocity_magnitude = 0.3  # Lateral velocity during scissor kicks
        self.roll_compensation = 0.09  # Reduced roll rate to stay within joint limits
        
        # Base foot positions (neutral stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (world frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Lateral velocity and roll compensation vary with scissor kick timing.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        if 0.0 <= phase < 0.2:
            # Right scissor kick: rightward motion with roll compensation
            progress = phase / 0.2
            # Smooth velocity profile using cosine-based envelope
            envelope = 0.5 * (1.0 - np.cos(np.pi * progress))
            vy = self.lateral_velocity_magnitude * np.sin(np.pi * progress)
            vz = 0.08 * np.sin(np.pi * progress)  # Increased vertical lift for ground clearance
            roll_rate = -self.roll_compensation * np.sin(np.pi * progress)
            
        elif 0.2 <= phase < 0.4:
            # Right return: smooth decay with improved damping
            progress = (phase - 0.2) / 0.2
            decay = np.cos(0.5 * np.pi * progress)
            vy = self.lateral_velocity_magnitude * 0.4 * decay
            vz = 0.04 * decay
            roll_rate = -self.roll_compensation * 0.2 * decay
            
        elif 0.4 <= phase < 0.6:
            # Left scissor kick: leftward motion with roll compensation
            progress = (phase - 0.4) / 0.2
            envelope = 0.5 * (1.0 - np.cos(np.pi * progress))
            vy = -self.lateral_velocity_magnitude * np.sin(np.pi * progress)
            vz = 0.08 * np.sin(np.pi * progress)  # Increased vertical lift for ground clearance
            roll_rate = self.roll_compensation * np.sin(np.pi * progress)
            
        elif 0.6 <= phase < 0.8:
            # Left return: smooth decay with improved damping
            progress = (phase - 0.6) / 0.2
            decay = np.cos(0.5 * np.pi * progress)
            vy = -self.lateral_velocity_magnitude * 0.4 * decay
            vz = 0.04 * decay
            roll_rate = self.roll_compensation * 0.2 * decay
            
        else:  # 0.8 <= phase < 1.0
            # Settle phase: smooth approach to zero using cosine taper
            progress = (phase - 0.8) / 0.2
            taper = np.cos(0.5 * np.pi * progress)
            vy = 0.0
            vz = 0.02 * taper
            roll_rate = 0.0
        
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
        Compute foot trajectory in body frame based on leg and phase.
        
        FL: neutral [0.0-0.4], upward kick [0.4-0.6], return/neutral [0.6-1.0]
        FR: upward kick [0.0-0.2], return/neutral [0.2-1.0]
        RL: neutral [0.0-0.4], limited downward push [0.4-0.6], return/neutral [0.6-1.0]
        RR: limited downward push [0.0-0.2], return/neutral [0.2-1.0]
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_name.startswith('FR'):
            if 0.0 <= phase < 0.2:
                # Upward scissor extension with smooth trajectory
                progress = phase / 0.2
                # Use smoothstep-like profile for phase-continuous motion
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                angle = np.pi * smooth_progress
                foot[2] += self.scissor_amplitude_up * np.sin(angle)
            elif 0.2 <= phase < 0.4:
                # Return to neutral with improved smoothness
                progress = (phase - 0.2) / 0.2
                # Exponential decay for smooth return
                decay = np.exp(-3.0 * progress)
                angle = np.pi * (1.0 - progress)
                foot[2] += self.scissor_amplitude_up * np.sin(angle) * decay * 0.2
            # else: remain at neutral
            
        elif leg_name.startswith('RR'):
            if 0.0 <= phase < 0.2:
                # Limited downward scissor extension to avoid ground penetration
                progress = phase / 0.2
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                angle = np.pi * smooth_progress
                foot[2] -= self.scissor_amplitude_down * np.sin(angle)
            elif 0.2 <= phase < 0.4:
                # Return to neutral with improved smoothness
                progress = (phase - 0.2) / 0.2
                decay = np.exp(-3.0 * progress)
                angle = np.pi * (1.0 - progress)
                foot[2] -= self.scissor_amplitude_down * np.sin(angle) * decay * 0.2
            # else: remain at neutral
            
        elif leg_name.startswith('FL'):
            if 0.4 <= phase < 0.6:
                # Upward scissor extension with smooth trajectory
                progress = (phase - 0.4) / 0.2
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                angle = np.pi * smooth_progress
                foot[2] += self.scissor_amplitude_up * np.sin(angle)
            elif 0.6 <= phase < 0.8:
                # Return to neutral with improved smoothness
                progress = (phase - 0.6) / 0.2
                decay = np.exp(-3.0 * progress)
                angle = np.pi * (1.0 - progress)
                foot[2] += self.scissor_amplitude_up * np.sin(angle) * decay * 0.2
            # else: remain at neutral (phases 0.0-0.4 and 0.8-1.0)
            
        elif leg_name.startswith('RL'):
            if 0.4 <= phase < 0.6:
                # Limited downward scissor extension to avoid ground penetration
                progress = (phase - 0.4) / 0.2
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                angle = np.pi * smooth_progress
                foot[2] -= self.scissor_amplitude_down * np.sin(angle)
            elif 0.6 <= phase < 0.8:
                # Return to neutral with improved smoothness
                progress = (phase - 0.6) / 0.2
                decay = np.exp(-3.0 * progress)
                angle = np.pi * (1.0 - progress)
                foot[2] -= self.scissor_amplitude_down * np.sin(angle) * decay * 0.2
            # else: remain at neutral (phases 0.0-0.4 and 0.8-1.0)
        
        return foot