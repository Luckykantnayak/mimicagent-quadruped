from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_CARTWHEEL_SIDE_TRAVEL_MotionGenerator(BaseMotionGenerator):
    """
    Lateral cartwheel motion with 360° roll and sideways travel.
    
    - Base performs continuous roll about longitudinal axis (360° per cycle)
    - Base translates laterally (positive y direction)
    - Right legs (FR, RR) serve as pivot during phase 0.0-0.5
    - Left legs (FL, RL) serve as pivot during phase 0.4-1.0
    - Overlap periods ensure continuous contact throughout cartwheel
    - Leg trajectories are circular arcs in body frame
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.4  # Slower frequency for controlled cartwheel rotation
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Cartwheel motion parameters
        self.roll_rate = 2.0 * np.pi * self.freq  # 360° per cycle
        self.lateral_velocity = 0.3  # Rightward travel speed (m/s)
        
        # Leg swing arc parameters
        self.swing_radius = 0.25  # Radius of circular arc for swing legs
        self.swing_height_max = 0.3  # Maximum height during overhead arc
        
        # Contact transition phases
        self.right_contact_start = 0.0
        self.right_contact_end = 0.5
        self.left_contact_start = 0.4
        self.left_contact_end = 1.0  # Wraps to next cycle
        
        # Transition blend width
        self.transition_blend = 0.1
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update_base_motion(self, phase, dt):
        """
        Update base with constant roll rate and lateral velocity.
        Roll rate integrates to 360° over full cycle.
        Lateral velocity produces rightward displacement.
        """
        # Constant positive roll rate throughout cycle
        roll_rate = self.roll_rate
        
        # Constant lateral velocity (positive y = rightward)
        vy = self.lateral_velocity
        
        self.vel_world = np.array([0.0, vy, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on contact/swing state.
        
        Right legs (FR, RR): stance 0.0-0.5, swing 0.5-1.0
        Left legs (FL, RL): swing 0.0-0.4, stance 0.4-1.0
        
        During stance: foot position rotates in body frame as base rolls
        During swing: foot traces circular arc overhead in body frame
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is left or right
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        if is_right:
            # Right legs: stance 0.0-0.5, swing 0.5-1.0
            if phase < 0.5:
                # Stance phase - simulate pivot point rotation
                foot = self._compute_stance_trajectory(base_pos, phase, 0.0, 0.5, is_left=False)
            else:
                # Swing phase - circular overhead arc
                foot = self._compute_swing_trajectory(base_pos, phase, 0.5, 1.0, is_left=False)
        
        elif is_left:
            # Left legs: swing 0.0-0.4, stance 0.4-1.0
            if phase < 0.4:
                # Swing phase - circular overhead arc
                foot = self._compute_swing_trajectory(base_pos, phase, 0.0, 0.4, is_left=True)
            else:
                # Stance phase - simulate pivot point rotation
                foot = self._compute_stance_trajectory(base_pos, phase, 0.4, 1.0, is_left=True)
        else:
            foot = base_pos
        
        return foot
    
    def _compute_stance_trajectory(self, base_pos, phase, start_phase, end_phase, is_left):
        """
        During stance, foot is planted in world frame but appears to move in body frame
        as the base rolls above it. Simulate this by rotating foot position about x-axis
        in body frame.
        """
        # Normalized progress through stance phase
        progress = (phase - start_phase) / (end_phase - start_phase)
        progress = np.clip(progress, 0.0, 1.0)
        
        # Foot rotates through 180° during its stance phase (half cartwheel)
        # Start from lower position, rotate through under-body to opposite side
        if is_left:
            # Left leg stance: starts from lower-left, rotates to upper-left
            angle_start = -np.pi / 4  # Lower left
            angle_end = 3 * np.pi / 4  # Upper left
        else:
            # Right leg stance: starts from lower-right, rotates to upper-right
            angle_start = -3 * np.pi / 4  # Lower right
            angle_end = np.pi / 4  # Upper right
        
        angle = angle_start + progress * (angle_end - angle_start)
        
        # Foot position rotates in y-z plane (roll about x-axis)
        foot = base_pos.copy()
        radius = np.sqrt(base_pos[1]**2 + base_pos[2]**2)
        foot[1] = radius * np.sin(angle)
        foot[2] = radius * np.cos(angle)
        
        return foot
    
    def _compute_swing_trajectory(self, base_pos, phase, start_phase, end_phase, is_left):
        """
        During swing, foot traces circular arc overhead in body frame.
        Arc moves from one side, up and over, to opposite side.
        """
        # Normalized progress through swing phase
        progress = (phase - start_phase) / (end_phase - start_phase)
        progress = np.clip(progress, 0.0, 1.0)
        
        # Swing arc spans 180° in y-z plane
        if is_left:
            # Left leg swing: from right-lower through overhead to left-lower
            angle_start = -np.pi / 3  # Right side, slightly up
            angle_end = np.pi + np.pi / 3  # Left side, slightly up
        else:
            # Right leg swing: from left-upper through overhead to right-lower
            angle_start = 2 * np.pi / 3  # Left side, upper
            angle_end = -np.pi / 3  # Right side, lower
        
        angle = angle_start + progress * (angle_end - angle_start)
        
        # Circular arc in y-z plane
        foot = base_pos.copy()
        
        # Swing radius and height
        swing_radius = 0.35
        
        # Parametric circle
        foot[1] = swing_radius * np.sin(angle)
        foot[2] = swing_radius * np.cos(angle) - (swing_radius - abs(base_pos[2]))
        
        # Add extra height during peak of swing (middle of arc)
        height_boost = self.swing_height_max * np.sin(np.pi * progress)
        foot[2] += height_boost
        
        return foot