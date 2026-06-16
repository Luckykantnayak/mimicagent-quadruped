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
        # Target: ~30° right roll, 30° forward pitch, 45° yaw by phase 0.25
        self.roll_rate_mag = 1.2  # rad/s
        self.pitch_rate_mag = 1.0  # rad/s
        self.yaw_rate_mag = 1.5    # rad/s
        
        # Foot adjustment amplitudes in body frame
        self.lateral_adjustment = 0.04   # Side-to-side adjustment for roll
        self.longitudinal_adjustment = 0.03  # Forward-backward for pitch
        self.vertical_adjustment = 0.02   # Vertical compensation

    def update_base_motion(self, phase, dt):
        """
        Update base angular velocities according to phase-dependent pretzel pattern.
        All linear velocities remain zero (in-place motion).
        """
        # Phase-dependent angular velocity commands
        if phase < 0.25:
            # Initial twist entry: pitch+, roll+, yaw+
            roll_rate = self.roll_rate_mag
            pitch_rate = self.pitch_rate_mag
            yaw_rate = self.yaw_rate_mag
            
        elif phase < 0.5:
            # Deep twist reversal: pitch+ (deepening), roll- (reverse), yaw+
            roll_rate = -self.roll_rate_mag * 1.3  # Stronger reversal
            pitch_rate = self.pitch_rate_mag * 1.2  # Deepening pitch
            yaw_rate = self.yaw_rate_mag
            
        elif phase < 0.75:
            # Asymmetric extension: pitch- (recovery), roll- (maximize left), yaw+
            roll_rate = -self.roll_rate_mag
            pitch_rate = -self.pitch_rate_mag * 1.1  # Recovery
            yaw_rate = self.yaw_rate_mag
            
        else:
            # Coordinated unwind: all return to neutral
            # Smooth unwinding by reversing accumulated rotations
            unwind_progress = (phase - 0.75) / 0.25
            roll_rate = self.roll_rate_mag * (1.0 - unwind_progress)
            pitch_rate = -self.pitch_rate_mag * 0.5 * (1.0 - unwind_progress)
            yaw_rate = self.yaw_rate_mag * 0.3 * (1.0 - unwind_progress)
        
        # Apply smooth transitions at phase boundaries using cosine blending
        blend_width = 0.05
        for boundary in [0.25, 0.5, 0.75]:
            if abs(phase - boundary) < blend_width:
                blend = 0.5 * (1.0 + np.cos(np.pi * (phase - boundary) / blend_width))
                # Smooth the rate change
                roll_rate *= (0.7 + 0.3 * blend)
                pitch_rate *= (0.7 + 0.3 * blend)
        
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

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with asymmetric adjustments to maintain
        contact and stability during 3D rotations.
        
        Feet adjust based on:
        - Roll direction: lateral shifts (left legs extend left during left roll)
        - Pitch direction: longitudinal shifts (front legs forward during forward pitch)
        - Yaw: diagonal weight redistribution
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg side and position
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Phase-dependent adjustments
        if phase < 0.25:
            # Initial twist: right roll, forward pitch
            # Right legs bear more load, extend right
            # Front legs bear more load, extend forward
            lateral_factor = -1.0 if is_left else 1.0
            longitudinal_factor = 1.0 if is_front else -1.0
            
            foot[1] += self.lateral_adjustment * lateral_factor * phase / 0.25
            foot[0] += self.longitudinal_adjustment * longitudinal_factor * phase / 0.25
            foot[2] -= self.vertical_adjustment * phase / 0.25
            
        elif phase < 0.5:
            # Deep twist reversal: roll reverses to left, pitch deepens
            # Left legs now bear more load
            local_phase = (phase - 0.25) / 0.25
            lateral_factor = 1.0 if is_left else -1.0
            longitudinal_factor = 1.0 if is_front else -1.0
            
            # Transition lateral adjustment
            roll_adjustment = self.lateral_adjustment * (1.0 - 2.0 * local_phase)
            foot[1] += roll_adjustment * (-1.0 if is_left else 1.0)
            
            # Increase forward pitch adjustment
            foot[0] += self.longitudinal_adjustment * longitudinal_factor * (1.0 + 0.5 * local_phase)
            foot[2] -= self.vertical_adjustment * (1.0 + 0.3 * local_phase)
            
        elif phase < 0.75:
            # Asymmetric extension: maximum left roll, pitch recovering
            local_phase = (phase - 0.5) / 0.25
            lateral_factor = 1.0 if is_left else -1.0
            
            # Maximum left roll adjustment
            foot[1] += self.lateral_adjustment * lateral_factor * 1.5
            
            # Pitch recovery: reduce longitudinal adjustment
            longitudinal_factor = 1.0 if is_front else -1.0
            foot[0] += self.longitudinal_adjustment * longitudinal_factor * (1.2 - 0.8 * local_phase)
            
            # Maintain vertical compensation
            foot[2] -= self.vertical_adjustment * (1.3 - 0.3 * local_phase)
            
        else:
            # Coordinated unwind: return to neutral
            local_phase = (phase - 0.75) / 0.25
            unwind = 1.0 - local_phase
            
            lateral_factor = 1.0 if is_left else -1.0
            longitudinal_factor = 1.0 if is_front else -1.0
            
            # Smoothly return all adjustments to zero
            foot[1] += self.lateral_adjustment * lateral_factor * 1.5 * unwind
            foot[0] += self.longitudinal_adjustment * longitudinal_factor * 0.4 * unwind
            foot[2] -= self.vertical_adjustment * unwind
        
        # Add subtle yaw compensation (diagonal pairs)
        is_diagonal_group1 = leg_name.startswith('FL') or leg_name.startswith('RR')
        yaw_phase = np.sin(2 * np.pi * phase)
        diagonal_factor = 1.0 if is_diagonal_group1 else -1.0
        foot[0] += 0.01 * diagonal_factor * yaw_phase
        foot[1] += 0.01 * diagonal_factor * yaw_phase
        
        return foot