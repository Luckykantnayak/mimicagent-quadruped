from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_COMPASS_NEEDLE_SEEK_MotionGenerator(BaseMotionGenerator):
    """
    Compass needle seek motion: damped in-place yaw oscillation.
    
    The robot performs alternating clockwise and counterclockwise rotations
    with progressively decreasing amplitude (40° → 30° → 20° → 12° → 5°),
    mimicking a compass needle settling onto a target heading.
    
    All four feet remain in continuous ground contact throughout.
    No linear translation occurs—only pure yaw rotation.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (body frame) - held constant throughout motion
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Oscillation sub-phases: [start, end, target_angle_deg, direction_sign]
        # direction_sign: +1 = clockwise, -1 = counterclockwise
        self.oscillation_phases = [
            {"range": [0.0, 0.2], "angle_deg": 40.0, "sign": 1.0},   # CW overshoot
            {"range": [0.2, 0.4], "angle_deg": 30.0, "sign": -1.0},  # CCW correction
            {"range": [0.4, 0.6], "angle_deg": 20.0, "sign": 1.0},   # CW medium
            {"range": [0.6, 0.8], "angle_deg": 12.0, "sign": -1.0},  # CCW small
            {"range": [0.8, 1.0], "angle_deg": 5.0, "sign": 1.0},    # CW final settle
        ]

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent yaw rate commands.
        Each sub-phase prescribes a target angular displacement over 0.2 phase duration.
        Linear velocities remain zero throughout (pure in-place rotation).
        """
        
        # Determine current sub-phase and compute required yaw rate
        yaw_rate = 0.0
        
        for osc in self.oscillation_phases:
            phase_start, phase_end = osc["range"]
            
            if phase_start <= phase < phase_end:
                # Target rotation in radians
                target_angle_rad = np.deg2rad(osc["angle_deg"])
                direction = osc["sign"]
                
                # Phase duration
                phase_duration = phase_end - phase_start
                
                # Time duration for this sub-phase at current frequency
                time_duration = phase_duration / self.freq
                
                # Constant yaw rate to achieve target angle over sub-phase duration
                # Positive yaw rate = clockwise rotation (right-hand rule: thumb down = CW when viewed from above)
                yaw_rate = direction * (target_angle_rad / time_duration)
                
                break
        
        # Set zero linear velocity (no translation)
        self.vel_world = np.array([0.0, 0.0, 0.0])
        
        # Set yaw rate (roll and pitch rates remain zero)
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
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
        All feet remain in continuous stance contact throughout the motion.
        Foot positions in body frame are held constant at their initial positions.
        
        The base rotation causes feet to appear to move in world frame,
        but in body frame they maintain nominal stance positions.
        """
        return self.base_feet_pos_body[leg_name].copy()