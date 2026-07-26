from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_KINETIC_CROSSOVER_SURF_MotionGenerator(BaseMotionGenerator):
    """
    Kinetic Crossover Surf: Skating-style locomotion with diagonal leg pairs 
    performing exaggerated crossover sweeps synchronized with rhythmic base 
    roll and yaw oscillations.
    
    - Continuous forward velocity maintained throughout
    - Diagonal pairs (FL+RR, FR+RL) alternate between inward crossover sweeps 
      and outward carving extensions
    - Base executes roll and yaw oscillations to create dynamic weight shifts
    - All four feet remain in ground contact (no flight phase)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Crossover cycle frequency (Hz)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.vx_forward = 0.8  # Constant forward velocity
        self.vy_amplitude = 0.3  # Lateral velocity oscillation amplitude
        self.yaw_rate_amplitude = 0.6  # Yaw rate oscillation amplitude (rad/s)
        self.roll_rate_amplitude = 0.8  # Roll rate oscillation amplitude (rad/s)
        
        # Leg crossover sweep parameters
        self.sweep_amplitude_lateral = 0.15  # Lateral sweep distance (m)
        self.sweep_amplitude_forward = 0.08  # Forward/rearward modulation during sweep
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Leg groupings: diagonal pairs
        self.group_1 = [name for name in leg_names if name.startswith('FL') or name.startswith('RR')]
        self.group_2 = [name for name in leg_names if name.startswith('FR') or name.startswith('RL')]

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity with sinusoidal lateral 
        velocity, roll rate, and yaw rate synchronized to crossover phases.
        
        Phase [0.0-0.5]: Left crossover (negative roll, negative yaw, left drift)
        Phase [0.5-1.0]: Right crossover (positive roll, positive yaw, right drift)
        """
        
        # Constant forward velocity
        vx = self.vx_forward
        
        # Lateral velocity: sinusoidal oscillation
        # Negative (left) during [0.0-0.5], positive (right) during [0.5-1.0]
        vy = self.vy_amplitude * np.sin(2 * np.pi * phase - np.pi / 2)
        
        # Roll rate: sinusoidal oscillation
        # Negative (left roll) during [0.0-0.25], positive (right roll) during [0.5-0.75]
        roll_rate = self.roll_rate_amplitude * np.sin(2 * np.pi * phase)
        
        # Yaw rate: sinusoidal oscillation with phase offset
        # Negative (CCW) during early left crossover, positive (CW) during early right crossover
        yaw_rate = self.yaw_rate_amplitude * np.sin(2 * np.pi * phase)
        
        # Set velocities in world frame
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, yaw_rate])
        
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
        Compute foot trajectory in body frame.
        
        Group 1 (FL, RR): Sweep inward during [0.0-0.5], outward during [0.5-1.0]
        Group 2 (FR, RL): Sweep outward during [0.0-0.5], inward during [0.5-1.0]
        
        All feet remain grounded (z ~ constant), with smooth lateral and 
        longitudinal arcs creating crossover motion.
        """
        
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is in group_1 or group_2
        is_group_1 = leg_name.startswith('FL') or leg_name.startswith('RR')
        
        # Determine if leg is front or rear
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Determine nominal lateral side (positive y = left, negative y = right)
        is_left_side = leg_name.startswith('FL') or leg_name.startswith('RL')
        side_sign = 1.0 if is_left_side else -1.0
        
        if is_group_1:
            # Group 1: inward sweep [0.0-0.5], outward sweep [0.5-1.0]
            if phase < 0.5:
                # Inward crossover sweep
                sweep_progress = phase / 0.5  # 0 -> 1
                # Smooth arc inward using cosine
                lateral_offset = -side_sign * self.sweep_amplitude_lateral * (1.0 - np.cos(np.pi * sweep_progress)) / 2.0
                # Forward modulation: advance slightly during sweep
                forward_offset = self.sweep_amplitude_forward * np.sin(np.pi * sweep_progress)
            else:
                # Outward recovery sweep
                sweep_progress = (phase - 0.5) / 0.5  # 0 -> 1
                # Smooth arc outward
                lateral_offset = -side_sign * self.sweep_amplitude_lateral * (1.0 + np.cos(np.pi * sweep_progress)) / 2.0
                # Rearward modulation: retract slightly during recovery
                forward_offset = -self.sweep_amplitude_forward * np.sin(np.pi * sweep_progress)
        else:
            # Group 2: outward sweep [0.0-0.5], inward sweep [0.5-1.0]
            if phase < 0.5:
                # Outward recovery sweep
                sweep_progress = phase / 0.5  # 0 -> 1
                # Smooth arc outward
                lateral_offset = side_sign * self.sweep_amplitude_lateral * (1.0 - np.cos(np.pi * sweep_progress)) / 2.0
                # Forward modulation
                forward_offset = self.sweep_amplitude_forward * np.sin(np.pi * sweep_progress)
            else:
                # Inward crossover sweep
                sweep_progress = (phase - 0.5) / 0.5  # 0 -> 1
                # Smooth arc inward
                lateral_offset = side_sign * self.sweep_amplitude_lateral * (1.0 + np.cos(np.pi * sweep_progress)) / 2.0
                # Rearward modulation
                forward_offset = -self.sweep_amplitude_forward * np.sin(np.pi * sweep_progress)
        
        # Apply offsets to base foot position
        foot = foot_base.copy()
        
        # Longitudinal (x): forward for front legs, rearward for rear legs
        if is_front:
            foot[0] += forward_offset
        else:
            foot[0] -= forward_offset
        
        # Lateral (y): apply computed lateral offset
        foot[1] += lateral_offset
        
        # Vertical (z): maintain ground contact with minimal vertical oscillation
        # Small vertical modulation for visual smoothness during crossover
        vertical_modulation = 0.01 * np.sin(2 * np.pi * phase)
        foot[2] += vertical_modulation
        
        return foot