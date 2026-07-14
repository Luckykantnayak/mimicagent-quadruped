from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_BARREL_ROLL_TRANSLATE_MotionGenerator(BaseMotionGenerator):
    """
    Barrel roll with forward translation.
    
    The robot performs a continuous 360-degree roll about its longitudinal axis
    while maintaining forward velocity. Legs tuck during inversion and extend
    for landing.
    
    Phase structure:
    - [0.0, 0.25]: Roll initiation, right tilt (0° → 90°), legs begin tucking
    - [0.25, 0.5]: Inverted transition (90° → 180°), all legs tucked
    - [0.5, 0.75]: Left tilt and recovery (180° → 270°), legs begin extending
    - [0.75, 1.0]: Upright restoration (270° → 360°), legs extend for landing
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0  # One complete barrel roll per cycle
        
        # Base foot positions (nominal stance in body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters
        self.forward_velocity = 0.8  # Constant forward velocity (m/s)
        self.roll_rate = 2 * np.pi  # 360° per cycle (rad/s at freq=1Hz)
        self.initial_upward_velocity = 0.4  # Launch impulse (m/s)
        self.landing_downward_velocity = 0.3  # Landing velocity (m/s)
        
        # Leg motion parameters
        self.tuck_radius = 0.15  # How close legs tuck to body center
        self.nominal_stance_radius = np.linalg.norm(self.base_feet_pos_body[leg_names[0]][:2])
        
        # Phase thresholds for leg motion
        self.phase_launch_end = 0.1
        self.phase_tuck_complete = 0.25
        self.phase_inverted_end = 0.5
        self.phase_extend_complete = 0.9
        self.phase_landing_start = 0.9

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity and roll rate.
        Vertical velocity varies by phase for launch and landing.
        """
        # Forward velocity is constant throughout
        vx = self.forward_velocity
        
        # Roll rate is constant throughout (360° over one cycle)
        roll_rate = self.roll_rate
        
        # Vertical velocity varies by phase
        if phase < self.phase_launch_end:
            # Initial launch: upward velocity for clearance
            progress = phase / self.phase_launch_end
            vz = self.initial_upward_velocity * (1.0 - progress)
        elif phase < self.phase_landing_start:
            # Airborne: no vertical velocity (ballistic)
            vz = 0.0
        else:
            # Landing approach: downward velocity
            progress = (phase - self.phase_landing_start) / (1.0 - self.phase_landing_start)
            vz = -self.landing_downward_velocity * progress
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
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
        Compute foot position in body frame based on phase.
        
        Legs tuck during roll and extend for landing.
        Right-side legs (FR, RR) tuck earlier; left-side legs (FL, RL) tuck later.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is on right or left side
        is_right_side = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Nominal stance position
        nominal_x = base_pos[0]
        nominal_y = base_pos[1]
        nominal_z = base_pos[2]
        
        # Compute tuck/extend state based on phase and side
        if phase < self.phase_launch_end:
            # Launch phase: legs in nominal stance
            tuck_factor = 0.0
            height_offset = 0.0
            
        elif phase < self.phase_tuck_complete:
            # Tucking phase: right side tucks faster
            progress = (phase - self.phase_launch_end) / (self.phase_tuck_complete - self.phase_launch_end)
            if is_right_side:
                # Right side tucks rapidly
                tuck_factor = progress
            else:
                # Left side begins tucking
                tuck_factor = progress * 0.7
            height_offset = 0.0
            
        elif phase < self.phase_inverted_end:
            # Inverted phase: all legs fully tucked
            tuck_factor = 1.0
            height_offset = 0.0
            
        elif phase < self.phase_extend_complete:
            # Extending phase: legs extend toward nominal
            progress = (phase - self.phase_inverted_end) / (self.phase_extend_complete - self.phase_inverted_end)
            if is_right_side:
                # Right side extends later
                tuck_factor = 1.0 - progress * 0.8
            else:
                # Left side extends earlier
                tuck_factor = 1.0 - progress
            height_offset = 0.0
            
        else:
            # Landing phase: legs fully extended to nominal stance
            progress = (phase - self.phase_extend_complete) / (1.0 - self.phase_extend_complete)
            tuck_factor = max(0.0, 1.0 - progress * 5.0)  # Quick final extension
            height_offset = 0.0
        
        # Apply tucking: interpolate between nominal and tucked position
        # Tucked position: closer to body center in horizontal plane, slightly raised in z
        tucked_x = nominal_x * (1.0 - tuck_factor * 0.7)
        tucked_y = nominal_y * (1.0 - tuck_factor * 0.7)
        tucked_z = nominal_z + tuck_factor * 0.1  # Slight upward tuck
        
        foot_pos = np.array([tucked_x, tucked_y, tucked_z + height_offset])
        
        return foot_pos