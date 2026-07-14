from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_METRONOME_SWAY_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Metronome sway advance gait with continuous forward locomotion.
    
    - Base rolls ±30° laterally in a cyclic pattern (right sway → neutral → left sway → neutral)
    - Forward velocity surges applied during neutral roll transitions
    - All four feet maintain continuous ground contact
    - Legs extend/compress to accommodate base roll while tracking ground
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for large roll amplitudes
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.roll_amplitude = np.deg2rad(30.0)  # ±30° roll
        self.lateral_displacement = 0.08  # Lateral sway amplitude (m)
        self.forward_surge_velocity = 0.6  # Forward velocity during neutral phases (m/s)
        self.forward_drift_velocity = 0.1  # Minimal forward drift during sway phases (m/s)
        
        # Leg extension parameters
        self.leg_extension_amplitude = 0.06  # Vertical extension/compression (m)
        self.lateral_tracking_amplitude = 0.03  # Lateral foot tracking (m)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent roll, lateral sway, and forward surges.
        
        Phase structure:
        [0.0, 0.25]: Right sway - roll to +30°, move right
        [0.25, 0.5]: Neutral surge 1 - unwind to 0°, surge forward
        [0.5, 0.75]: Left sway - roll to -30°, move left
        [0.75, 1.0]: Neutral surge 2 - unwind to 0°, surge forward
        """
        
        # Compute sub-phase and local progress
        if phase < 0.25:
            # Right sway phase
            sub_phase = 0
            local_progress = phase / 0.25
            target_roll = self.roll_amplitude * self._smooth_step(local_progress)
            roll_rate = self._compute_derivative(0, 0.25, phase, self.roll_amplitude)
            vx = self.forward_drift_velocity
            vy = self.lateral_displacement / (0.25 / self.freq) * np.cos(2 * np.pi * local_progress)
            
        elif phase < 0.5:
            # Neutral surge 1 phase
            sub_phase = 1
            local_progress = (phase - 0.25) / 0.25
            target_roll = self.roll_amplitude * (1.0 - self._smooth_step(local_progress))
            roll_rate = self._compute_derivative(0.25, 0.5, phase, -self.roll_amplitude)
            vx = self.forward_surge_velocity
            vy = -self.lateral_displacement / (0.25 / self.freq) * np.cos(2 * np.pi * local_progress)
            
        elif phase < 0.75:
            # Left sway phase
            sub_phase = 2
            local_progress = (phase - 0.5) / 0.25
            target_roll = -self.roll_amplitude * self._smooth_step(local_progress)
            roll_rate = self._compute_derivative(0.5, 0.75, phase, -self.roll_amplitude)
            vx = self.forward_drift_velocity
            vy = -self.lateral_displacement / (0.25 / self.freq) * np.cos(2 * np.pi * local_progress)
            
        else:
            # Neutral surge 2 phase
            sub_phase = 3
            local_progress = (phase - 0.75) / 0.25
            target_roll = -self.roll_amplitude * (1.0 - self._smooth_step(local_progress))
            roll_rate = self._compute_derivative(0.75, 1.0, phase, self.roll_amplitude)
            vx = self.forward_surge_velocity
            vy = self.lateral_displacement / (0.25 / self.freq) * np.cos(2 * np.pi * local_progress)
        
        # Set velocity commands (WORLD frame)
        self.vel_world = np.array([vx, vy, 0.0])
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
        Compute foot position in body frame with vertical extension/compression
        and lateral tracking to maintain ground contact during roll.
        
        Left legs (FL, RL): extend during right sway, compress during left sway
        Right legs (FR, RR): compress during right sway, extend during left sway
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is on left or right side
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compute current roll phase function (-1 to +1)
        # Positive = right sway, Negative = left sway
        if phase < 0.25:
            # Right sway: 0 → +1
            roll_phase = self._smooth_step(phase / 0.25)
        elif phase < 0.5:
            # Neutral 1: +1 → 0
            roll_phase = 1.0 - self._smooth_step((phase - 0.25) / 0.25)
        elif phase < 0.75:
            # Left sway: 0 → -1
            roll_phase = -self._smooth_step((phase - 0.5) / 0.25)
        else:
            # Neutral 2: -1 → 0
            roll_phase = -(1.0 - self._smooth_step((phase - 0.75) / 0.25))
        
        # Vertical extension/compression
        # Left legs: extend when roll_phase > 0 (right sway), compress when roll_phase < 0 (left sway)
        # Right legs: compress when roll_phase > 0 (right sway), extend when roll_phase < 0 (left sway)
        if is_left_leg:
            z_offset = -self.leg_extension_amplitude * roll_phase
        else:
            z_offset = self.leg_extension_amplitude * roll_phase
        
        foot[2] += z_offset
        
        # Lateral tracking to compensate for base lateral shift
        # When base moves right (+y), feet shift right to maintain body-relative position
        lateral_shift = self.lateral_tracking_amplitude * roll_phase
        foot[1] += lateral_shift
        
        return foot

    def _smooth_step(self, t):
        """
        Smooth step function using cubic Hermite interpolation.
        Input t ∈ [0, 1], Output ∈ [0, 1] with zero derivatives at endpoints.
        """
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _compute_derivative(self, phase_start, phase_end, phase, amplitude):
        """
        Compute smooth derivative (rate) for a transition over a phase interval.
        Uses derivative of smooth_step scaled by amplitude and phase duration.
        """
        phase_duration = phase_end - phase_start
        local_progress = (phase - phase_start) / phase_duration
        local_progress = np.clip(local_progress, 0.0, 1.0)
        
        # Derivative of smooth_step: 6t(1-t)
        derivative_local = 6.0 * local_progress * (1.0 - local_progress)
        
        # Scale by amplitude and account for phase duration (convert to rate per second)
        rate = amplitude * derivative_local / (phase_duration / self.freq)
        
        return rate