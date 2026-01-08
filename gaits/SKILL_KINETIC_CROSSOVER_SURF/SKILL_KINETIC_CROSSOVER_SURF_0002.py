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
        self.roll_amplitude = 0.12  # Roll angle oscillation amplitude (rad) - bounded
        
        # Leg crossover sweep parameters
        self.sweep_amplitude_lateral = 0.15  # Lateral sweep distance (m)
        self.sweep_amplitude_forward = 0.08  # Forward/rearward modulation during sweep
        
        # Vertical compensation parameters
        self.base_height_lift = 0.04  # Additional baseline clearance (m)
        self.roll_compensation_gain = 1.0  # Multiplier for roll-based vertical compensation
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.current_roll = 0.0  # Track current roll angle
        
        # Leg groupings: diagonal pairs
        self.group_1 = [name for name in leg_names if name.startswith('FL') or name.startswith('RR')]
        self.group_2 = [name for name in leg_names if name.startswith('FR') or name.startswith('RL')]

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity with sinusoidal lateral 
        velocity, bounded roll angle, and yaw rate synchronized to crossover phases.
        
        Phase [0.0-0.5]: Left crossover (negative roll, negative yaw, left drift)
        Phase [0.5-1.0]: Right crossover (positive roll, positive yaw, right drift)
        """
        
        # Constant forward velocity
        vx = self.vx_forward
        
        # Lateral velocity: sinusoidal oscillation
        vy = self.vy_amplitude * np.sin(2 * np.pi * phase - np.pi / 2)
        
        # Roll angle: bounded sinusoidal oscillation (not integrated from rate)
        target_roll = self.roll_amplitude * np.sin(2 * np.pi * phase)
        
        # Yaw rate: sinusoidal oscillation
        yaw_rate = self.yaw_rate_amplitude * np.sin(2 * np.pi * phase)
        
        # Compute roll rate from target roll for smooth integration
        roll_rate = self.roll_amplitude * 2 * np.pi * self.freq * np.cos(2 * np.pi * phase)
        
        # Set velocities in world frame
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, yaw_rate])
        
        # Update current roll for foot compensation
        self.current_roll = target_roll
        
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
        Compute foot trajectory in body frame with roll compensation.
        
        Group 1 (FL, RR): Sweep inward during [0.0-0.5], outward during [0.5-1.0]
        Group 2 (FR, RL): Sweep outward during [0.0-0.5], inward during [0.5-1.0]
        
        All feet remain grounded with vertical compensation for base roll.
        """
        
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg properties
        is_group_1 = leg_name.startswith('FL') or leg_name.startswith('RR')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_left_side = leg_name.startswith('FL') or leg_name.startswith('RL')
        side_sign = 1.0 if is_left_side else -1.0
        
        # Compute crossover sweep offsets
        if is_group_1:
            # Group 1: inward sweep [0.0-0.5], outward sweep [0.5-1.0]
            if phase < 0.5:
                sweep_progress = phase / 0.5
                lateral_offset = -side_sign * self.sweep_amplitude_lateral * (1.0 - np.cos(np.pi * sweep_progress)) / 2.0
                forward_offset = self.sweep_amplitude_forward * np.sin(np.pi * sweep_progress)
            else:
                sweep_progress = (phase - 0.5) / 0.5
                lateral_offset = -side_sign * self.sweep_amplitude_lateral * (1.0 + np.cos(np.pi * sweep_progress)) / 2.0
                forward_offset = -self.sweep_amplitude_forward * np.sin(np.pi * sweep_progress)
        else:
            # Group 2: outward sweep [0.0-0.5], inward sweep [0.5-1.0]
            if phase < 0.5:
                sweep_progress = phase / 0.5
                lateral_offset = side_sign * self.sweep_amplitude_lateral * (1.0 - np.cos(np.pi * sweep_progress)) / 2.0
                forward_offset = self.sweep_amplitude_forward * np.sin(np.pi * sweep_progress)
            else:
                sweep_progress = (phase - 0.5) / 0.5
                lateral_offset = side_sign * self.sweep_amplitude_lateral * (1.0 + np.cos(np.pi * sweep_progress)) / 2.0
                forward_offset = -self.sweep_amplitude_forward * np.sin(np.pi * sweep_progress)
        
        # Apply offsets to base foot position
        foot = foot_base.copy()
        
        # Longitudinal (x)
        if is_front:
            foot[0] += forward_offset
        else:
            foot[0] -= forward_offset
        
        # Lateral (y)
        foot[1] += lateral_offset
        
        # Vertical (z): roll compensation + baseline lift + smooth oscillation
        
        # Base height lift for all legs
        vertical_offset = self.base_height_lift
        
        # Roll compensation: counteract body roll effect on ground clearance
        # When body rolls left (negative), right legs need more height
        # When body rolls right (positive), left legs need more height
        # The lateral offset of the foot determines how much roll affects its height
        foot_lateral_position = foot[1]
        roll_induced_height_change = -self.current_roll * foot_lateral_position * self.roll_compensation_gain
        vertical_offset += roll_induced_height_change
        
        # Phase-synchronous additional compensation for safety
        # Right-side legs get extra lift during left roll (phase 0.0-0.5)
        # Left-side legs get extra lift during right roll (phase 0.5-1.0)
        phase_compensation = 0.0
        if not is_left_side:  # Right-side legs
            # Add lift during left roll (negative roll, phase near 0.25)
            phase_compensation = 0.02 * max(0.0, -np.sin(2 * np.pi * phase))
        else:  # Left-side legs
            # Add lift during right roll (positive roll, phase near 0.75)
            phase_compensation = 0.02 * max(0.0, np.sin(2 * np.pi * phase))
        
        vertical_offset += phase_compensation
        
        # Smooth vertical oscillation for visual flow
        vertical_oscillation = 0.015 * np.sin(2 * np.pi * phase)
        vertical_offset += vertical_oscillation
        
        foot[2] += vertical_offset
        
        return foot