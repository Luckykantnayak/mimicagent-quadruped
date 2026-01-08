from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CYCLONE_WHEEL_SLASH_MotionGenerator(BaseMotionGenerator):
    """
    Cyclone Wheel Slash: High-speed spinning maneuver with dynamic lateral drift.
    
    All four wheels maintain ground contact throughout. The cyclone effect is achieved
    through aggressive yaw rotation combined with lateral velocity modulation and
    diagonal leg extension/retraction patterns that modulate traction dynamically.
    
    Phase structure:
        [0.0, 0.25]: spin_initiation - rapid yaw acceleration with forward bias
        [0.25, 0.5]: peak_cyclone - maximum yaw rate with lateral slash
        [0.5, 0.75]: slash_transition - controlled deceleration with direction change
        [0.75, 1.0]: burst_reacceleration - explosive re-acceleration burst
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz) - tunable for dramatic vs rapid slashes
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Leg extension parameters
        self.extension_amplitude = 0.04  # ~10-15% modulation of leg length
        self.retraction_amplitude = 0.02  # Reduced pressure during pivot
        
        # Base velocity parameters
        self.vx_max = 0.8  # Forward velocity magnitude (m/s)
        self.vy_amplitude = 0.5  # Lateral drift amplitude (m/s)
        self.yaw_rate_max = 4.5  # Peak yaw rate (rad/s) - aggressive but stable
        
        # State variables
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Diagonal pairing for coordination
        # Group 1: FL-RR (in-phase)
        # Group 2: FR-RL (in-phase, out-of-phase with Group 1)
        self.group_1 = [leg for leg in leg_names if leg.startswith('FL') or leg.startswith('RR')]
        self.group_2 = [leg for leg in leg_names if leg.startswith('FR') or leg.startswith('RL')]

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Implements aggressive yaw spinning with coordinated lateral drift.
        """
        # Forward velocity profile
        if phase < 0.25:
            # Spin initiation: rapidly increasing forward velocity
            progress = phase / 0.25
            vx = self.vx_max * progress
        elif phase < 0.5:
            # Peak cyclone: sustained forward velocity
            vx = self.vx_max
        elif phase < 0.75:
            # Slash transition: decreasing forward velocity
            progress = (phase - 0.5) / 0.25
            vx = self.vx_max * (1.0 - 0.6 * progress)
        else:
            # Burst reacceleration: rapid increase
            progress = (phase - 0.75) / 0.25
            vx = self.vx_max * (0.4 + 0.8 * progress)
        
        # Lateral velocity profile (creates the "slash" effect)
        if phase < 0.25:
            # Initiation: slight rightward bias building
            progress = phase / 0.25
            vy = 0.3 * self.vy_amplitude * progress
        elif phase < 0.5:
            # Peak cyclone: lateral reversal from right to left
            progress = (phase - 0.25) / 0.25
            vy = 0.3 * self.vy_amplitude * (1.0 - 2.0 * progress) - self.vy_amplitude * progress
        elif phase < 0.75:
            # Transition: sustained leftward drift
            vy = -0.7 * self.vy_amplitude
        else:
            # Burst: transition back toward neutral/slight right
            progress = (phase - 0.75) / 0.25
            vy = -0.7 * self.vy_amplitude * (1.0 - progress) + 0.2 * self.vy_amplitude * progress
        
        # Yaw rate profile (creates the cyclone rotation)
        if phase < 0.25:
            # Spin initiation: rapid ramp-up
            progress = phase / 0.25
            yaw_rate = self.yaw_rate_max * np.sin(0.5 * np.pi * progress)
        elif phase < 0.5:
            # Peak cyclone: sustained maximum yaw rate
            yaw_rate = self.yaw_rate_max
        elif phase < 0.75:
            # Transition: controlled deceleration
            progress = (phase - 0.5) / 0.25
            yaw_rate = self.yaw_rate_max * (1.0 - 0.7 * progress)
        else:
            # Burst: sharp re-acceleration
            progress = (phase - 0.75) / 0.25
            yaw_rate = self.yaw_rate_max * (0.3 + 0.9 * np.sin(0.5 * np.pi * progress))
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate base pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in BODY frame with dynamic extension/retraction.
        
        Diagonal coordination:
        - Group 1 (FL, RR): in-phase modulation
        - Group 2 (FR, RL): in-phase modulation, out-of-phase with Group 1
        
        All feet remain in contact (stance) throughout.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine diagonal group membership
        is_group_1 = leg_name.startswith('FL') or leg_name.startswith('RR')
        
        # Compute leg extension/retraction based on phase
        if phase < 0.25:
            # Spin initiation: all legs extend for maximum traction
            progress = phase / 0.25
            extension = -self.extension_amplitude * np.sin(0.5 * np.pi * progress)
        elif phase < 0.5:
            # Peak cyclone: diagonal modulation creates dynamic weight shifting
            progress = (phase - 0.25) / 0.25
            modulation_angle = 2.0 * np.pi * progress
            if is_group_1:
                # Group 1 modulation
                extension = -self.extension_amplitude * (0.8 + 0.4 * np.sin(modulation_angle))
            else:
                # Group 2 modulation (out-of-phase)
                extension = -self.extension_amplitude * (0.8 - 0.4 * np.sin(modulation_angle))
        elif phase < 0.75:
            # Transition: slight retraction for pivot facilitation
            progress = (phase - 0.5) / 0.25
            extension = -self.extension_amplitude * (1.0 - progress) - self.retraction_amplitude * progress
        else:
            # Burst: rapid re-extension for acceleration
            progress = (phase - 0.75) / 0.25
            extension = -self.retraction_amplitude * (1.0 - progress) - self.extension_amplitude * progress
        
        # Apply vertical extension (z-axis in body frame is down for feet)
        foot[2] += extension
        
        # Subtle forward/backward shifting during phases for enhanced traction control
        if phase < 0.5:
            # Forward bias during spin-up and peak
            foot[0] += 0.01 * np.sin(np.pi * phase / 0.5)
        else:
            # Slight rearward shift during transition and burst
            foot[0] -= 0.01 * np.sin(np.pi * (phase - 0.5) / 0.5)
        
        return foot