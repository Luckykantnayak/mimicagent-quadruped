from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PRETZEL_TWIST_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Pretzel Twist Roll: Complex in-place 3D rotational maneuver combining
    simultaneous pitch, roll, and yaw angular velocities in a coordinated pattern.
    
    Four sub-phases:
    - [0.0, 0.25]: Initial twist entry (pitch+, roll+, yaw+)
    - [0.25, 0.5]: Deep twist reversal (pitch+, roll-, yaw+)
    - [0.5, 0.75]: Asymmetric extension (pitch-, roll-, yaw+)
    - [0.75, 1.0]: Coordinated unwind (all return to neutral)
    
    All four feet remain in contact throughout, repositioning in body frame
    to maintain stability as base orientation changes.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.4  # Slower frequency for complex motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Angular velocity magnitudes (tuned for target orientations)
        self.roll_rate_mag = 1.0  # rad/s
        self.pitch_rate_mag = 0.9  # rad/s
        self.yaw_rate_mag = 1.3    # rad/s
        
        # Foot adjustment amplitudes in body frame
        self.lateral_adjustment = 0.03   # Side-to-side adjustment for roll
        self.longitudinal_adjustment = 0.025  # Forward-backward for pitch
        self.vertical_adjustment = 0.04   # Vertical compensation (increased for better ground contact)

    def update_base_motion(self, phase, dt):
        """
        Update base angular velocities according to phase-dependent pretzel pattern.
        All linear velocities remain zero (in-place motion).
        """
        # Smooth phase transitions using cosine interpolation
        def smooth_transition(p, boundary, width=0.08):
            if abs(p - boundary) < width:
                return 0.5 * (1.0 + np.cos(np.pi * (p - boundary) / width))
            return 1.0
        
        # Phase-dependent angular velocity commands with smooth transitions
        if phase < 0.25:
            # Initial twist entry: pitch+, roll+, yaw+
            progress = phase / 0.25
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            roll_rate = self.roll_rate_mag * smooth_progress
            pitch_rate = self.pitch_rate_mag * smooth_progress
            yaw_rate = self.yaw_rate_mag * smooth_progress
            
        elif phase < 0.5:
            # Deep twist reversal: pitch+ (deepening), roll- (reverse), yaw+
            progress = (phase - 0.25) / 0.25
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            # Smooth transition from positive to negative roll
            roll_rate = self.roll_rate_mag * (1.0 - 2.0 * smooth_progress) * 1.2
            pitch_rate = self.pitch_rate_mag * (1.0 + 0.2 * smooth_progress)
            yaw_rate = self.yaw_rate_mag * (1.0 + 0.1 * smooth_progress)
            
        elif phase < 0.75:
            # Asymmetric extension: pitch- (recovery), roll- (maximize left), yaw+
            progress = (phase - 0.5) / 0.25
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            roll_rate = -self.roll_rate_mag * (1.2 + 0.3 * smooth_progress)
            pitch_rate = self.pitch_rate_mag * (1.2 - 2.2 * smooth_progress)
            yaw_rate = self.yaw_rate_mag * (1.1 - 0.2 * smooth_progress)
            
        else:
            # Coordinated unwind: all return to neutral
            progress = (phase - 0.75) / 0.25
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            unwind = 1.0 - smooth_progress
            
            roll_rate = -self.roll_rate_mag * 0.8 * unwind
            pitch_rate = -self.pitch_rate_mag * 0.7 * unwind
            yaw_rate = self.yaw_rate_mag * 0.4 * unwind
        
        # Set velocities (world frame)
        self.vel_world = np.array([0.0, 0.0, 0.0])  # In-place motion
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Adjust root height to maintain ground contact
        self.adjust_root_height_for_ground_contact()

    def adjust_root_height_for_ground_contact(self):
        """
        Adjust root z-position to ensure at least one foot maintains ground contact
        and prevent all feet from being airborne simultaneously.
        """
        # Transform all feet to world frame
        foot_positions_world = {}
        for leg_name in self.leg_names:
            foot_body = self.current_foot_positions_body.get(leg_name, self.base_feet_pos_body[leg_name])
            foot_world = quat_rotate(self.root_quat, foot_body) + self.root_pos
            foot_positions_world[leg_name] = foot_world
        
        # Find the minimum z-coordinate among all feet
        min_z = min(foot_world[2] for foot_world in foot_positions_world.values())
        
        # Adjust root position to bring lowest foot to ground level (z=0)
        if min_z > 0:
            self.root_pos[2] -= min_z
        elif min_z < -0.005:
            # Allow slight negative (compression) but limit penetration
            self.root_pos[2] -= (min_z + 0.005)

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with leg-specific vertical compensation
        to counteract geometric effects of 3D rotation and maintain ground contact.
        
        Key principle: When a leg is geometrically lowered by body rotation (e.g., front-right
        during forward pitch + right roll), compensate by RAISING it in body frame.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg geometry
        is_left = leg_name in ['FL', 'RL']
        is_right = leg_name in ['FR', 'RR']
        is_front = leg_name in ['FL', 'FR']
        is_rear = leg_name in ['RL', 'RR']
        
        # Leg-specific vertical compensation based on rotation geometry
        def get_vertical_compensation(phase):
            """
            Calculate leg-specific vertical adjustment based on which legs are
            geometrically lowered/raised by the current phase's rotation combination.
            """
            if phase < 0.25:
                # Forward pitch + right roll: FR and RL corners drop, FL and RR rise
                if leg_name == 'FR':
                    return self.vertical_adjustment * 1.5 * (phase / 0.25)
                elif leg_name == 'RL':
                    return self.vertical_adjustment * 1.5 * (phase / 0.25)
                elif leg_name == 'FL':
                    return self.vertical_adjustment * 0.3 * (phase / 0.25)
                elif leg_name == 'RR':
                    return self.vertical_adjustment * 0.3 * (phase / 0.25)
                    
            elif phase < 0.5:
                # Deepening forward pitch + left roll: FL and RR drop, FR and RL rise
                local_progress = (phase - 0.25) / 0.25
                if leg_name == 'FL':
                    return self.vertical_adjustment * (1.5 + 0.8 * local_progress)
                elif leg_name == 'RR':
                    return self.vertical_adjustment * (1.5 + 0.8 * local_progress)
                elif leg_name == 'FR':
                    return self.vertical_adjustment * (1.5 - 0.9 * local_progress)
                elif leg_name == 'RL':
                    return self.vertical_adjustment * (1.5 - 0.9 * local_progress)
                    
            elif phase < 0.75:
                # Pitch recovering + maximum left roll: FL and RR very low, FR and RL higher
                local_progress = (phase - 0.5) / 0.25
                if leg_name == 'FL':
                    return self.vertical_adjustment * (2.3 - 0.3 * local_progress)
                elif leg_name == 'RR':
                    return self.vertical_adjustment * (2.3 - 0.3 * local_progress)
                elif leg_name == 'FR':
                    return self.vertical_adjustment * (0.6 + 0.2 * local_progress)
                elif leg_name == 'RL':
                    return self.vertical_adjustment * (0.6 + 0.2 * local_progress)
                    
            else:
                # Unwind: smooth return to neutral
                local_progress = (phase - 0.75) / 0.25
                unwind = 1.0 - local_progress
                smooth_unwind = 0.5 * (1.0 + np.cos(np.pi * (1.0 - unwind)))
                
                if leg_name == 'FL' or leg_name == 'RR':
                    return self.vertical_adjustment * 2.0 * smooth_unwind
                else:
                    return self.vertical_adjustment * 0.8 * smooth_unwind
            
            return 0.0
        
        # Apply vertical compensation (UPWARD for geometrically-lowered legs)
        foot[2] += get_vertical_compensation(phase)
        
        # Lateral adjustments for roll compensation
        if phase < 0.25:
            # Right roll: right legs extend right, left legs extend left slightly
            progress = phase / 0.25
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            if is_right:
                foot[1] += self.lateral_adjustment * smooth_progress * 0.8
            else:
                foot[1] -= self.lateral_adjustment * smooth_progress * 0.5
                
        elif phase < 0.5:
            # Transitioning to left roll
            local_progress = (phase - 0.25) / 0.25
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * local_progress))
            roll_factor = 1.0 - 2.0 * smooth_progress
            if is_right:
                foot[1] += self.lateral_adjustment * 0.8 * roll_factor
            else:
                foot[1] -= self.lateral_adjustment * 0.5 * roll_factor
                
        elif phase < 0.75:
            # Maximum left roll: left legs extend left, right legs extend right
            local_progress = (phase - 0.5) / 0.25
            if is_left:
                foot[1] -= self.lateral_adjustment * (1.2 + 0.3 * local_progress)
            else:
                foot[1] += self.lateral_adjustment * (1.0 + 0.2 * local_progress)
                
        else:
            # Unwind lateral
            local_progress = (phase - 0.75) / 0.25
            unwind = 1.0 - local_progress
            smooth_unwind = 0.5 * (1.0 + np.cos(np.pi * (1.0 - unwind)))
            if is_left:
                foot[1] -= self.lateral_adjustment * 1.5 * smooth_unwind
            else:
                foot[1] += self.lateral_adjustment * 1.2 * smooth_unwind
        
        # Longitudinal adjustments for pitch compensation
        if phase < 0.5:
            # Forward pitch: front legs move slightly forward, rear slightly back
            progress = phase / 0.5
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            if is_front:
                foot[0] += self.longitudinal_adjustment * smooth_progress * 0.6
            else:
                foot[0] -= self.longitudinal_adjustment * smooth_progress * 0.5
        else:
            # Pitch recovering: reduce longitudinal offset
            local_progress = (phase - 0.5) / 0.5
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * local_progress))
            unwind = 1.0 - smooth_progress
            if is_front:
                foot[0] += self.longitudinal_adjustment * 0.6 * unwind
            else:
                foot[0] -= self.longitudinal_adjustment * 0.5 * unwind
        
        # Subtle yaw compensation (diagonal redistribution)
        yaw_phase = np.sin(2 * np.pi * phase) * 0.5
        is_diagonal_1 = leg_name in ['FL', 'RR']
        diagonal_factor = 1.0 if is_diagonal_1 else -1.0
        foot[0] += 0.008 * diagonal_factor * yaw_phase
        foot[1] += 0.008 * diagonal_factor * yaw_phase
        
        return foot