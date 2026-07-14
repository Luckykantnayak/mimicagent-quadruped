from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Ricochet bounce diagonal gait with alternating left/right diagonal bounces.
    
    Motion cycle:
    - Phase 0.0-0.2: Left compression with positive yaw
    - Phase 0.2-0.4: Right diagonal launch (aerial)
    - Phase 0.4-0.6: Right compression with negative yaw
    - Phase 0.6-0.8: Left diagonal launch (aerial)
    - Phase 0.8-1.0: Landing preparation with yaw neutralization
    
    Contact pattern alternates between left-side (FL, RL) and right-side (FR, RR) diagonal pairs.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency for ricochet pattern
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Target nominal height for regulation
        self.nominal_height = 0.45
        self.height_regulation_gain = 1.5
        
        # Motion parameters - reduced from original to respect kinematic constraints
        self.forward_velocity = 0.8  # Sustained forward velocity
        self.base_compression_velocity = -0.6  # Reduced downward velocity during compression
        self.base_launch_velocity = 1.2  # Reduced upward velocity during launch
        self.launch_lateral_velocity = 0.8  # Lateral velocity during diagonal launch
        self.yaw_rate_magnitude = 2.5  # Yaw rotation rate during compression (rad/s)
        
        # Leg motion parameters - reduced to prevent joint limit violations
        self.stance_compression = 0.03  # Feet stay near ground during stance, body lowers
        self.swing_clearance = 0.08  # Vertical clearance during swing phase
        self.forward_reach = 0.08  # Forward reach during stance preparation
        self.lateral_spread = 0.03  # Lateral foot adjustment during stance

    def update_base_motion(self, phase, dt):
        """
        Update base motion with phase-dependent velocity commands and height regulation.
        Implements alternating compression/launch pattern with yaw rotation.
        """
        vx = self.forward_velocity
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Height regulation: adjust vertical velocity to maintain nominal height
        height_error = self.nominal_height - self.root_pos[2]
        height_correction = height_error * self.height_regulation_gain
        
        # Phase 0.0-0.2: Left compression with positive yaw
        if 0.0 <= phase < 0.2:
            t_local = (phase - 0.0) / 0.2
            # Smooth compression with sinusoidal envelope
            compression_factor = np.sin(t_local * np.pi)
            vz = self.base_compression_velocity * compression_factor + height_correction * 0.3
            yaw_rate = self.yaw_rate_magnitude * (1.0 - t_local * 0.5)
        
        # Phase 0.2-0.4: Right diagonal launch (aerial)
        elif 0.2 <= phase < 0.4:
            t_local = (phase - 0.2) / 0.2
            # Smooth launch with velocity tapering at end
            launch_factor = 1.0 - 0.4 * t_local
            vz = self.base_launch_velocity * launch_factor + height_correction * 0.5
            vy = self.launch_lateral_velocity * (1.0 - t_local * 0.3)
            yaw_rate = -self.yaw_rate_magnitude * t_local * 0.5
        
        # Phase 0.4-0.6: Right compression with negative yaw
        elif 0.4 <= phase < 0.6:
            t_local = (phase - 0.4) / 0.2
            # Smooth compression with sinusoidal envelope
            compression_factor = np.sin(t_local * np.pi)
            vz = self.base_compression_velocity * compression_factor + height_correction * 0.3
            yaw_rate = -self.yaw_rate_magnitude * (1.0 - t_local * 0.5)
        
        # Phase 0.6-0.8: Left diagonal launch (aerial)
        elif 0.6 <= phase < 0.8:
            t_local = (phase - 0.6) / 0.2
            # Smooth launch with velocity tapering at end
            launch_factor = 1.0 - 0.4 * t_local
            vz = self.base_launch_velocity * launch_factor + height_correction * 0.5
            vy = -self.launch_lateral_velocity * (1.0 - t_local * 0.3)
            yaw_rate = self.yaw_rate_magnitude * t_local * 0.5
        
        # Phase 0.8-1.0: Landing preparation with yaw neutralization
        else:
            t_local = (phase - 0.8) / 0.2
            # Gentle descent with smooth transition
            vz = self.base_compression_velocity * 0.6 * np.sin(t_local * np.pi) + height_correction * 0.4
            yaw_rate = self.yaw_rate_magnitude * 0.3 * (1.0 - t_local)
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for each leg based on phase.
        Implements alternating diagonal contact pattern with compression/extension cycles.
        Feet remain near ground during stance; body compression achieved via base motion.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Left-side legs: FL, RL (stance during 0.0-0.2 and 0.6-1.0)
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            is_front = leg_name.startswith('FL')
            
            # Phase 0.0-0.2: Stance with compression (feet stay grounded)
            if 0.0 <= phase < 0.2:
                t_local = phase / 0.2
                # Minimal vertical adjustment - feet remain planted
                foot[2] -= self.stance_compression * np.sin(t_local * np.pi)
                # Forward positioning for stance
                foot[0] += self.forward_reach * (1.0 - 0.3 * t_local) if is_front else -self.forward_reach * 0.5
                foot[1] -= self.lateral_spread * (1.0 - 0.2 * t_local)
            
            # Phase 0.2-0.4: Swing with clearance (right pair in stance)
            elif 0.2 <= phase < 0.4:
                t_local = (phase - 0.2) / 0.2
                # Lift foot for clearance during swing
                clearance_profile = np.sin(t_local * np.pi)
                foot[2] += self.swing_clearance * clearance_profile
                # Advance foot forward during swing
                foot[0] += self.forward_reach * (0.7 + 0.3 * t_local) if is_front else -self.forward_reach * 0.3
                foot[1] -= self.lateral_spread * (1.0 - t_local * 0.5)
            
            # Phase 0.4-0.6: Continue swing (right pair in stance)
            elif 0.4 <= phase < 0.6:
                t_local = (phase - 0.4) / 0.2
                # Maintain clearance, prepare for landing
                clearance_profile = np.sin((0.5 + t_local * 0.5) * np.pi)
                foot[2] += self.swing_clearance * 0.6 * clearance_profile
                # Position for upcoming stance
                foot[0] += self.forward_reach * (1.0 - 0.2 * t_local) if is_front else -self.forward_reach * 0.4
                foot[1] -= self.lateral_spread * 0.5
            
            # Phase 0.6-0.8: Transition to stance during left launch
            elif 0.6 <= phase < 0.8:
                t_local = (phase - 0.6) / 0.2
                # Lower foot to ground for stance
                foot[2] += self.swing_clearance * 0.3 * (1.0 - t_local)
                # Establish stance position
                foot[0] += self.forward_reach * (0.8 + 0.2 * t_local) if is_front else -self.forward_reach * 0.5
                foot[1] -= self.lateral_spread * (0.5 + 0.5 * t_local)
            
            # Phase 0.8-1.0: Early stance with landing preparation
            else:
                t_local = (phase - 0.8) / 0.2
                # Feet grounded for landing absorption
                foot[2] -= self.stance_compression * 0.5 * np.sin(t_local * np.pi)
                foot[0] += self.forward_reach if is_front else -self.forward_reach * 0.5
                foot[1] -= self.lateral_spread
        
        # Right-side legs: FR, RR (stance during 0.2-0.6)
        else:
            is_front = leg_name.startswith('FR')
            
            # Phase 0.0-0.2: Swing with clearance (left pair in stance)
            if 0.0 <= phase < 0.2:
                t_local = phase / 0.2
                # Lift foot for clearance during swing
                clearance_profile = np.sin(t_local * np.pi)
                foot[2] += self.swing_clearance * clearance_profile
                # Advance foot forward during swing
                foot[0] += self.forward_reach * (0.7 + 0.3 * t_local) if is_front else -self.forward_reach * 0.3
                foot[1] += self.lateral_spread * (1.0 - t_local * 0.5)
            
            # Phase 0.2-0.4: Transition to stance during right launch
            elif 0.2 <= phase < 0.4:
                t_local = (phase - 0.2) / 0.2
                # Lower foot to ground for stance
                foot[2] += self.swing_clearance * 0.3 * (1.0 - t_local)
                # Establish stance position
                foot[0] += self.forward_reach * (0.8 + 0.2 * t_local) if is_front else -self.forward_reach * 0.5
                foot[1] += self.lateral_spread * (0.5 + 0.5 * t_local)
            
            # Phase 0.4-0.6: Stance with compression (feet stay grounded)
            elif 0.4 <= phase < 0.6:
                t_local = (phase - 0.4) / 0.2
                # Minimal vertical adjustment - feet remain planted
                foot[2] -= self.stance_compression * np.sin(t_local * np.pi)
                # Forward positioning for stance
                foot[0] += self.forward_reach * (1.0 - 0.3 * t_local) if is_front else -self.forward_reach * 0.5
                foot[1] += self.lateral_spread * (1.0 - 0.2 * t_local)
            
            # Phase 0.6-0.8: Swing with clearance (left pair in stance)
            elif 0.6 <= phase < 0.8:
                t_local = (phase - 0.6) / 0.2
                # Lift foot for clearance during swing
                clearance_profile = np.sin(t_local * np.pi)
                foot[2] += self.swing_clearance * clearance_profile
                # Advance foot forward during swing
                foot[0] += self.forward_reach * (0.7 + 0.3 * t_local) if is_front else -self.forward_reach * 0.3
                foot[1] += self.lateral_spread * (1.0 - t_local * 0.5)
            
            # Phase 0.8-1.0: Continue swing during landing preparation
            else:
                t_local = (phase - 0.8) / 0.2
                # Maintain clearance, prepare for next stance
                clearance_profile = np.sin((0.5 + t_local * 0.5) * np.pi)
                foot[2] += self.swing_clearance * 0.6 * clearance_profile
                # Position for upcoming stance
                foot[0] += self.forward_reach * (1.0 - 0.2 * t_local) if is_front else -self.forward_reach * 0.4
                foot[1] += self.lateral_spread * 0.5
        
        return foot