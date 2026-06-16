from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_METRONOME_SWAY_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Metronome sway advance gait: forward locomotion via large-amplitude lateral swaying.
    
    - Base performs sinusoidal roll oscillation (±30°) like an inverted pendulum
    - Forward velocity surges occur during neutral roll transitions
    - Lateral velocity alternates left-right with roll direction
    - All four feet maintain continuous ground contact
    - Leg compression alternates between left and right pairs to support weight shifts
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slow metronome-like sway frequency
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.max_roll_angle = np.deg2rad(30.0)  # ±30° roll amplitude
        self.forward_surge_speed = 0.4  # Peak forward velocity during neutral phases
        self.lateral_sway_speed = 0.3  # Peak lateral velocity during roll phases
        self.compression_range = 0.06  # Leg compression/extension range (meters)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (world frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with sinusoidal roll oscillation and phase-dependent velocities.
        
        Phase structure:
        - [0.0-0.25]: Roll right, minimal forward, rightward lateral
        - [0.25-0.5]: Pass through neutral, forward surge, lateral reverses
        - [0.5-0.75]: Roll left, minimal forward, leftward lateral
        - [0.75-1.0]: Pass through neutral, forward surge, lateral reverses
        """
        
        # Roll angle: sinusoidal oscillation with period 1.0
        # phase 0 → 0°, phase 0.25 → +30°, phase 0.5 → 0°, phase 0.75 → -30°, phase 1.0 → 0°
        target_roll = self.max_roll_angle * np.sin(2.0 * np.pi * phase)
        
        # Roll rate: derivative of roll angle
        roll_rate = 2.0 * np.pi * self.freq * self.max_roll_angle * np.cos(2.0 * np.pi * phase)
        
        # Forward velocity: peaks during neutral phases [0.25-0.5] and [0.75-1.0]
        # Use cosine squared envelope to create surges at quarter and three-quarter phases
        forward_envelope = np.cos(2.0 * np.pi * phase) ** 2  # Peaks at 0.25 and 0.75
        vx = self.forward_surge_speed * forward_envelope
        
        # Lateral velocity: follows roll direction, reverses at phase 0.5
        # Positive (right) during [0-0.5], negative (left) during [0.5-1.0]
        # Magnitude peaks during maximum roll phases, minimizes during neutral
        lateral_envelope = np.abs(np.sin(2.0 * np.pi * phase))  # Peaks at 0.25 and 0.75
        vy = self.lateral_sway_speed * lateral_envelope * np.sign(np.cos(2.0 * np.pi * phase))
        
        # Vertical velocity: slight lowering during max roll, raising during neutral
        # Inverse of lateral envelope
        vz = -0.05 * lateral_envelope
        
        # Set world-frame velocities
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame with differential leg compression.
        
        Leg behavior:
        - Right legs (FR, RR): compress during right roll [0-0.25], extend during left roll [0.5-0.75]
        - Left legs (FL, RL): extend during right roll [0-0.25], compress during left roll [0.5-0.75]
        - All legs equalize during neutral transitions [0.25-0.5] and [0.75-1.0]
        
        Feet remain stationary in world frame (continuous contact), so body-frame
        z-coordinate varies to reflect leg length changes as base rolls.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is on left or right side
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compression function: sinusoidal variation synchronized with roll
        # Right legs: compress when roll is positive (phase ~0.25), extend when roll is negative (phase ~0.75)
        # Left legs: extend when roll is positive (phase ~0.25), compress when roll is negative (phase ~0.75)
        roll_phase = np.sin(2.0 * np.pi * phase)  # +1 at phase 0.25, -1 at phase 0.75
        
        if is_right_leg:
            # Right legs compress during positive roll, extend during negative roll
            compression = -self.compression_range * roll_phase
        elif is_left_leg:
            # Left legs extend during positive roll, compress during negative roll
            compression = self.compression_range * roll_phase
        else:
            compression = 0.0
        
        # Apply vertical compression/extension
        # Positive compression → foot moves up in body frame (leg shortens)
        # Negative compression → foot moves down in body frame (leg lengthens)
        foot[2] += compression
        
        # Small lateral shift to maintain contact during roll
        # As base rolls, feet shift slightly in body frame to compensate
        if is_right_leg:
            foot[1] += 0.02 * roll_phase  # Shift right when rolling right
        elif is_left_leg:
            foot[1] -= 0.02 * roll_phase  # Shift left when rolling right
        
        return foot