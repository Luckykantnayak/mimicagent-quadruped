from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CARVE_AND_CROSS_SLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Skating-style carving motion with alternating left and right diagonal leans.
    
    - Phase [0.0-0.5]: Left carve (yaw left, roll right, FL/RL grip, FR/RR slide)
    - Phase [0.5-1.0]: Right carve (yaw right, roll left, FR/RR grip, FL/RL slide)
    - All four wheels remain in contact throughout
    - Base motion driven by velocity and angular rate commands
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for smooth carving arcs
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Add safety margin to initial foot positions to prevent immediate penetration
        for leg_name in self.base_feet_pos_body:
            self.base_feet_pos_body[leg_name][2] += 0.08  # Lift feet 8cm higher in body frame
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.vx_forward = 0.8  # Forward velocity baseline
        self.vy_lateral_peak = 0.6  # Peak lateral velocity during carve
        self.base_height_drop = 0.04  # Gentle base height drop during peak carve (reduced from 0.15)
        
        self.yaw_rate_peak = 1.2  # Peak yaw rate (rad/s)
        self.roll_rate_peak = 0.6  # Peak roll rate (reduced from 0.8 to limit tilt)
        
        # Stance width modulation
        self.stance_width_grip = 0.12  # Additional lateral offset for gripping edge
        self.stance_width_slide = -0.08  # Reduced lateral offset for sliding side
        self.stance_lift_slide = 0.015  # Slight lift for sliding side to reduce contact pressure

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent linear and angular velocities.
        Left carve [0.0-0.5], Right carve [0.5-1.0]
        Staggered activation: roll builds first, then lateral velocity
        """
        
        # Determine carve side and local phase within carve
        if phase < 0.5:
            # Left carve
            carve_phase = phase * 2.0  # Map [0.0-0.5] to [0.0-1.0]
            carve_sign = -1.0  # Left carve: negative yaw, positive roll
        else:
            # Right carve
            carve_phase = (phase - 0.5) * 2.0  # Map [0.5-1.0] to [0.0-1.0]
            carve_sign = 1.0  # Right carve: positive yaw, negative roll
        
        # Compute intensity profiles with smoother transitions
        if carve_phase < 0.25:
            # Transition: ramp up roll first
            roll_intensity = carve_phase / 0.25
            lateral_intensity = 0.0
            height_intensity = 0.0
        elif carve_phase < 0.4:
            # Entry: roll at peak, lateral velocity ramping
            roll_intensity = 1.0
            lateral_intensity = (carve_phase - 0.25) / 0.15
            height_intensity = (carve_phase - 0.25) / 0.15
        elif carve_phase < 0.7:
            # Hold: sustained peak
            roll_intensity = 1.0
            lateral_intensity = 1.0
            height_intensity = 1.0
        elif carve_phase < 0.85:
            # Exit: ramp down lateral first
            roll_intensity = 1.0
            lateral_intensity = 1.0 - (carve_phase - 0.7) / 0.15
            height_intensity = 1.0 - (carve_phase - 0.7) / 0.15
        else:
            # Final transition: ramp down roll
            roll_intensity = 1.0 - (carve_phase - 0.85) / 0.15
            lateral_intensity = 0.0
            height_intensity = 0.0
        
        # Apply smoothing function
        roll_smooth = 3 * roll_intensity**2 - 2 * roll_intensity**3
        lateral_smooth = 3 * lateral_intensity**2 - 2 * lateral_intensity**3
        height_smooth = 3 * height_intensity**2 - 2 * height_intensity**3
        
        # Linear velocities
        vx = self.vx_forward
        vy = carve_sign * self.vy_lateral_peak * lateral_smooth
        
        # Base height modulation: gentle drop during hold phase only
        vz = -self.base_height_drop * height_smooth
        
        self.vel_world = np.array([vx, vy, vz])
        
        # Angular velocities with staggered activation
        yaw_rate = carve_sign * self.yaw_rate_peak * lateral_smooth
        roll_rate = -carve_sign * self.roll_rate_peak * roll_smooth  # Opposite sign to yaw
        pitch_rate = 0.0
        
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
        Compute foot position in body frame with stance width modulation.
        Gripping diagonal widens stance laterally.
        Sliding diagonal narrows stance and lifts slightly to reduce contact.
        No downward extension to avoid ground penetration during roll.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine which diagonal this leg belongs to
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Determine carve side and local phase
        if phase < 0.5:
            # Left carve: FL/RL grip (widen), FR/RR slide (narrow + lift)
            carve_phase = phase * 2.0
            if is_left_leg:
                role = 'grip'
            else:
                role = 'slide'
        else:
            # Right carve: FR/RR grip (widen), FL/RL slide (narrow + lift)
            carve_phase = (phase - 0.5) * 2.0
            if is_right_leg:
                role = 'grip'
            else:
                role = 'slide'
        
        # Compute intensity for stance modulation with smooth transitions
        if carve_phase < 0.3:
            intensity = carve_phase / 0.3
        elif carve_phase < 0.7:
            intensity = 1.0
        else:
            intensity = 1.0 - (carve_phase - 0.7) / 0.3
        
        intensity_smooth = 3 * intensity**2 - 2 * intensity**3
        
        # Apply stance width modulation
        if role == 'grip':
            # Widen stance laterally for gripping edge
            lateral_offset = self.stance_width_grip * intensity_smooth
            if is_left_leg:
                foot[1] -= lateral_offset  # Move left leg further left
            else:
                foot[1] += lateral_offset  # Move right leg further right
            
            # No downward extension - rely on base roll for geometric loading
            
        else:
            # Narrow stance and lift slightly for sliding side
            lateral_offset = self.stance_width_slide * intensity_smooth
            if is_left_leg:
                foot[1] -= lateral_offset  # Move left leg toward center
            else:
                foot[1] += lateral_offset  # Move right leg toward center
            
            # Slight upward lift to reduce contact pressure on sliding side
            foot[2] += self.stance_lift_slide * intensity_smooth
        
        return foot