from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERSE_CARTWHEEL_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Reverse cartwheel drift: continuous backward motion with 360-degree roll per cycle.
    
    - Base executes full roll rotation (360 degrees) over one phase cycle
    - Continuous backward velocity (negative vx) throughout
    - Left legs (FL, RL) provide stance during phase [0, 0.25], then swing overhead
    - Right legs (FR, RR) swing overhead during phase [0, 0.5], then provide stance
    - Inverted aerial phase around [0.25, 0.5]
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for complex cartwheel motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Cartwheel motion parameters
        self.backward_velocity = -0.8  # Sustained backward drift
        self.roll_rate_base = 2.0 * np.pi  # 360 degrees per cycle (adjusted by freq)
        
        # Leg extension parameters
        self.leg_extension_radius = 0.35  # Radial extension during overhead swing
        self.stance_width = 0.15  # Lateral offset during stance
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base with continuous backward velocity and roll rotation.
        Roll rate varies slightly across phases for smooth transitions.
        """
        # Backward velocity with slight variation (peak during inverted phase)
        if 0.25 <= phase < 0.5:
            vx = self.backward_velocity * 1.2  # Peak during aerial/inverted
        else:
            vx = self.backward_velocity
        
        # Roll rate: sustained positive rotation, slightly modulated for smoothness
        if phase < 0.25:
            # Initial ascent: increasing roll rate
            roll_rate = self.roll_rate_base * (0.8 + 0.4 * (phase / 0.25))
        elif phase < 0.5:
            # Inverted transition: peak roll rate
            roll_rate = self.roll_rate_base * 1.2
        elif phase < 0.75:
            # Left ascent: sustained roll rate
            roll_rate = self.roll_rate_base * 1.0
        else:
            # Upright return: decreasing for smooth cycle
            roll_rate = self.roll_rate_base * (1.0 - 0.2 * ((phase - 0.75) / 0.25))
        
        # Minor lateral drift based on roll phase for realism
        vy = 0.05 * np.sin(2 * np.pi * phase)
        
        # Vertical velocity near zero (slight variation during transitions)
        vz = 0.02 * np.cos(4 * np.pi * phase)
        
        # Set world frame velocities
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])  # Pure roll rotation
        
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
        Compute foot position in body frame based on cartwheel phase.
        
        Left legs (FL, RL): stance [0, 0.25], swing overhead [0.25, 1.0]
        Right legs (FR, RR): swing overhead [0, 0.5], stance [0.5, 1.0]
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        if is_left:
            # Left legs: stance [0, 0.25], swing [0.25, 1.0]
            foot = self._compute_left_leg_trajectory(base_pos, phase, is_front)
        else:
            # Right legs: swing [0, 0.5], stance [0.5, 1.0]
            foot = self._compute_right_leg_trajectory(base_pos, phase, is_front)
        
        return foot

    def _compute_left_leg_trajectory(self, base_pos, phase, is_front):
        """
        Left leg trajectory: stance [0, 0.25], overhead swing [0.25, 1.0]
        """
        foot = base_pos.copy()
        
        if phase < 0.25:
            # Stance phase: maintain ground contact with backward sweep
            progress = phase / 0.25
            foot[0] = base_pos[0] + 0.08 * (0.5 - progress)  # Backward sweep
            foot[1] = -self.stance_width  # Left side
            foot[2] = base_pos[2]  # Ground level
            
        elif phase < 0.5:
            # Ascending overhead: extend radially away from body
            progress = (phase - 0.25) / 0.25
            angle = np.pi * progress  # 0 to 180 degrees
            
            # Circular arc from ground to overhead
            foot[0] = base_pos[0] + 0.1 * np.cos(angle)
            foot[1] = -self.leg_extension_radius * np.cos(angle * 0.5)
            foot[2] = self.leg_extension_radius * np.sin(angle)
            
        elif phase < 0.75:
            # Fully overhead: maximum extension
            progress = (phase - 0.5) / 0.25
            angle = np.pi + np.pi * progress  # 180 to 360 degrees
            
            foot[0] = base_pos[0] + 0.1 * np.cos(angle)
            foot[1] = -self.leg_extension_radius * 0.3
            foot[2] = self.leg_extension_radius * (1.0 - 0.3 * progress)
            
        else:
            # Descending to ground: prepare for next stance
            progress = (phase - 0.75) / 0.25
            angle = np.pi * (1.0 - progress)  # 180 to 0 degrees
            
            foot[0] = base_pos[0] + 0.08 * progress
            foot[1] = -self.stance_width * (1.0 + 0.5 * (1.0 - progress))
            foot[2] = self.leg_extension_radius * np.sin(angle) * 0.5
        
        return foot

    def _compute_right_leg_trajectory(self, base_pos, phase, is_front):
        """
        Right leg trajectory: overhead swing [0, 0.5], stance [0.5, 1.0]
        """
        foot = base_pos.copy()
        
        if phase < 0.25:
            # Ascending overhead: extend radially
            progress = phase / 0.25
            angle = np.pi * progress  # 0 to 180 degrees
            
            foot[0] = base_pos[0] + 0.1 * np.cos(angle)
            foot[1] = self.leg_extension_radius * np.cos(angle * 0.5)
            foot[2] = self.leg_extension_radius * np.sin(angle)
            
        elif phase < 0.5:
            # Fully overhead during inverted phase
            progress = (phase - 0.25) / 0.25
            angle = np.pi + np.pi * progress  # 180 to 360 degrees
            
            foot[0] = base_pos[0] + 0.1 * np.cos(angle)
            foot[1] = self.leg_extension_radius * 0.3
            foot[2] = self.leg_extension_radius * (1.0 - 0.3 * progress)
            
        elif phase < 0.75:
            # Descending to stance: establish ground contact
            progress = (phase - 0.5) / 0.25
            angle = np.pi * (1.0 - progress)  # 180 to 0 degrees
            
            foot[0] = base_pos[0] + 0.08 * (0.5 - progress)
            foot[1] = self.stance_width
            foot[2] = self.leg_extension_radius * np.sin(angle) * 0.5
            
        else:
            # Stance phase: maintain ground contact with backward sweep
            progress = (phase - 0.75) / 0.25
            foot[0] = base_pos[0] + 0.08 * (0.5 - progress)  # Backward sweep
            foot[1] = self.stance_width  # Right side
            foot[2] = base_pos[2]  # Ground level
        
        return foot