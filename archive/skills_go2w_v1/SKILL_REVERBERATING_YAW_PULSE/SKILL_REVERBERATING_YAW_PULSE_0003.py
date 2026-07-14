from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERBERATING_YAW_PULSE_MotionGenerator(BaseMotionGenerator):
    """
    Reverberating Yaw Pulse Skill: Damped oscillatory yaw rotation settling to 45° CCW.
    
    Motion consists of five alternating yaw pulses with decreasing amplitude:
    - Phase [0.0, 0.2]: 60° CCW
    - Phase [0.2, 0.4]: 40° CW
    - Phase [0.4, 0.6]: 25° CCW
    - Phase [0.6, 0.8]: 15° CW
    - Phase [0.8, 1.0]: 5° CCW
    Net result: ~45° CCW rotation with in-place stance.
    
    All four feet maintain ground contact throughout. Legs extend radially during
    CCW pulses to increase moment arm, retract during CW reversals to reduce inertia.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Phase boundaries for five pulses
        self.phase_boundaries = [
            (0.0, 0.2),   # Primary pulse CCW
            (0.2, 0.4),   # First reversal CW
            (0.4, 0.6),   # Second reversal CCW
            (0.6, 0.8),   # Third reversal CW
            (0.8, 1.0)    # Final settling CCW
        ]
        
        # Yaw rate parameters (rad/s) calibrated to achieve target angular displacements
        # Target rotations: 60°, -40°, 25°, -15°, 5° over 0.2 phase duration each
        # Assuming freq=1.0 Hz → total cycle time = 1.0s → each phase = 0.2s
        # Required yaw_rate = target_angle / phase_duration
        self.yaw_rates = [
            np.deg2rad(60) / 0.2,    # 300 deg/s = 5.236 rad/s CCW
            -np.deg2rad(40) / 0.2,   # -200 deg/s = -3.491 rad/s CW
            np.deg2rad(25) / 0.2,    # 125 deg/s = 2.182 rad/s CCW
            -np.deg2rad(15) / 0.2,   # -75 deg/s = -1.309 rad/s CW
            np.deg2rad(5) / 0.2      # 25 deg/s = 0.436 rad/s CCW
        ]
        
        # Leg extension parameters: radial displacement amplitudes (m)
        # Legs extend during CCW pulses (positive yaw rate), retract during CW
        self.extension_amplitudes = [
            0.08,   # Primary pulse: strong extension
            -0.05,  # First reversal: moderate retraction
            0.04,   # Second reversal: moderate extension
            -0.02,  # Third reversal: small retraction
            0.01    # Final settling: minimal extension
        ]

    def update_base_motion(self, phase, dt):
        """
        Update base angular velocity based on current phase.
        All linear velocities remain zero to keep robot in place.
        """
        # Determine which pulse phase we're in
        yaw_rate = 0.0
        for i, (p_start, p_end) in enumerate(self.phase_boundaries):
            if p_start <= phase < p_end:
                yaw_rate = self.yaw_rates[i]
                break
        
        # Zero linear velocity (in-place rotation)
        self.vel_world = np.array([0.0, 0.0, 0.0])
        
        # Pure yaw rotation (no roll or pitch)
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
        Compute foot position in body frame with radial extension/retraction.
        
        During CCW pulses: feet extend radially outward to increase moment arm
        During CW reversals: feet retract inward to reduce rotational inertia
        
        Extension is applied radially from body center in the xy-plane.
        """
        # Get base foot position
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        # Determine current pulse phase and extension amplitude
        extension_amp = 0.0
        for i, (p_start, p_end) in enumerate(self.phase_boundaries):
            if p_start <= phase < p_end:
                extension_amp = self.extension_amplitudes[i]
                # Smooth interpolation within phase using cosine taper
                local_phase = (phase - p_start) / (p_end - p_start)
                # Use smooth transition: ramp up first half, ramp down second half
                smooth_factor = np.sin(np.pi * local_phase)
                extension_amp *= smooth_factor
                break
        
        # Compute radial direction in xy-plane from body center
        xy_dist = np.sqrt(foot_base[0]**2 + foot_base[1]**2)
        if xy_dist > 1e-6:
            radial_dir = np.array([foot_base[0], foot_base[1], 0.0]) / xy_dist
        else:
            radial_dir = np.array([1.0, 0.0, 0.0])
        
        # Apply radial extension
        foot = foot_base + extension_amp * radial_dir
        
        return foot