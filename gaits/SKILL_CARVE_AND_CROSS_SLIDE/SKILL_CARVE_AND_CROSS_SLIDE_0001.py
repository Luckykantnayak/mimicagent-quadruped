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
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.vx_forward = 0.8  # Forward velocity baseline
        self.vy_lateral_peak = 0.6  # Peak lateral velocity during carve
        self.vz_drop = -0.15  # Base height drop during peak carve
        
        self.yaw_rate_peak = 1.2  # Peak yaw rate (rad/s)
        self.roll_rate_peak = 0.8  # Peak roll rate (rad/s)
        
        # Stance width modulation
        self.stance_width_grip = 0.12  # Additional lateral offset for gripping edge
        self.stance_width_slide = -0.08  # Reduced lateral offset for sliding side
        self.stance_height_drop = 0.05  # Leg extension during peak carve

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent linear and angular velocities.
        Left carve [0.0-0.5], Right carve [0.5-1.0]
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
        
        # Compute intensity profiles across carve phase
        # Transition in [0.0-0.2], Entry [0.2-0.5], Hold [0.5-0.8], Exit [0.8-1.0]
        if carve_phase < 0.2:
            # Transition: ramp up from zero
            intensity = carve_phase / 0.2
        elif carve_phase < 0.5:
            # Entry: ramp to peak
            intensity = 0.5 + 0.5 * (carve_phase - 0.2) / 0.3
        elif carve_phase < 0.8:
            # Hold: sustained peak
            intensity = 1.0
        else:
            # Exit: ramp down
            intensity = 1.0 - (carve_phase - 0.8) / 0.2
        
        # Smoothing function for transitions
        intensity_smooth = 3 * intensity**2 - 2 * intensity**3
        
        # Linear velocities
        vx = self.vx_forward
        vy = carve_sign * self.vy_lateral_peak * intensity_smooth
        
        # Base height modulation: drop during peak carve (intensity high)
        if carve_phase >= 0.5 and carve_phase < 0.8:
            vz = self.vz_drop * 0.5  # Maintain low height
        elif carve_phase >= 0.2 and carve_phase < 0.5:
            vz = self.vz_drop * 2.0  # Drop during entry
        elif carve_phase >= 0.8:
            vz = -self.vz_drop * 2.0  # Rise during exit
        else:
            vz = 0.0  # Neutral during transition
        
        self.vel_world = np.array([vx, vy, vz])
        
        # Angular velocities
        yaw_rate = carve_sign * self.yaw_rate_peak * intensity_smooth
        roll_rate = -carve_sign * self.roll_rate_peak * intensity_smooth  # Opposite sign to yaw
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
        Gripping diagonal widens stance, sliding diagonal narrows stance.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine which diagonal this leg belongs to
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Determine carve side and local phase
        if phase < 0.5:
            # Left carve: FL/RL grip (widen), FR/RR slide (narrow)
            carve_phase = phase * 2.0
            if is_left_leg:
                # Gripping edge
                role = 'grip'
            else:
                # Sliding side
                role = 'slide'
        else:
            # Right carve: FR/RR grip (widen), FL/RL slide (narrow)
            carve_phase = (phase - 0.5) * 2.0
            if is_right_leg:
                # Gripping edge
                role = 'grip'
            else:
                # Sliding side
                role = 'slide'
        
        # Compute intensity for stance modulation
        if carve_phase < 0.2:
            intensity = carve_phase / 0.2
        elif carve_phase < 0.8:
            intensity = 1.0
        else:
            intensity = 1.0 - (carve_phase - 0.8) / 0.2
        
        intensity_smooth = 3 * intensity**2 - 2 * intensity**3
        
        # Apply stance width modulation
        if role == 'grip':
            # Widen stance laterally for gripping edge
            lateral_offset = self.stance_width_grip * intensity_smooth
            if is_left_leg:
                foot[1] -= lateral_offset  # Move left leg further left
            else:
                foot[1] += lateral_offset  # Move right leg further right
            
            # Extend leg downward slightly to maintain contact during roll
            foot[2] -= self.stance_height_drop * intensity_smooth
            
        else:
            # Narrow stance for sliding side
            lateral_offset = self.stance_width_slide * intensity_smooth
            if is_left_leg:
                foot[1] -= lateral_offset  # Move left leg toward center
            else:
                foot[1] += lateral_offset  # Move right leg toward center
        
        return foot