from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PHOENIX_RISE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Phoenix Rise Spin: A dramatic rising motion from crouch to maximum height
    while executing a full 360-degree yaw rotation with radially symmetric
    leg extension creating a 'wings spreading' aesthetic.
    
    All four feet remain in ground contact throughout the motion.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.4  # Slow dramatic motion, ~2.5 seconds per cycle
        
        # Base foot positions (nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.max_height_gain = 0.3  # Maximum vertical rise in meters
        self.total_yaw_rotation = 2 * np.pi  # 360 degrees in radians
        self.max_radial_extension = 0.25  # Maximum outward foot displacement
        self.max_upward_foot_displacement = 0.1  # Upward foot motion during extension
        
        # Phase boundaries for motion stages
        self.phase_crouch_end = 0.2
        self.phase_rise_start = 0.2
        self.phase_mid_rise = 0.4
        self.phase_peak_approach = 0.6
        self.phase_peak_start = 0.8
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Store initial height for relative motion
        self.initial_height = 0.0
        self.target_height = 0.0

    def reset(self, root_pos, root_quat):
        """Override reset to store initial height reference"""
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0
        self.initial_height = root_pos[2]
        self.target_height = self.initial_height + self.max_height_gain

    def update_base_motion(self, phase, dt):
        """
        Update base motion with vertical velocity and yaw rotation.
        
        Phase 0.0-0.2: Stationary crouch
        Phase 0.2-0.8: Rising with constant yaw rate
        Phase 0.8-1.0: Hold height, complete rotation
        """
        
        # Compute smooth vertical velocity profile
        if phase < self.phase_crouch_end:
            # Initial crouch: no motion
            vz = 0.0
        elif phase < self.phase_peak_start:
            # Rising phase: smooth acceleration and deceleration
            rise_phase = (phase - self.phase_rise_start) / (self.phase_peak_start - self.phase_rise_start)
            # Use smooth bell curve (sine-based) for velocity profile
            vz_scale = np.sin(rise_phase * np.pi)
            # Total rise duration
            rise_duration = (self.phase_peak_start - self.phase_rise_start) / self.freq
            # Average velocity needed
            avg_vz = self.max_height_gain / rise_duration
            vz = avg_vz * vz_scale * 1.5  # Scale by 1.5 to compensate for sine average
        else:
            # Peak hold: no vertical motion
            vz = 0.0
        
        # Compute yaw rate (constant during rotation phases)
        if phase < self.phase_rise_start:
            # No rotation during initial crouch
            yaw_rate = 0.0
        else:
            # Constant yaw rate from phase 0.2 to 1.0
            rotation_duration = (1.0 - self.phase_rise_start) / self.freq
            yaw_rate = self.total_yaw_rotation / rotation_duration
        
        # Set velocity commands
        self.vel_world = np.array([0.0, 0.0, vz])
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
        Compute foot position in body frame with radial extension.
        
        All legs extend radially outward and slightly upward as the body rises,
        creating a symmetric 'wing spreading' motion. Feet remain in contact
        (Z trajectory keeps feet at ground level relative to extending legs).
        """
        
        # Get base foot position
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial direction from body center (in XY plane)
        radial_xy = np.array([base_foot[0], base_foot[1]])
        radial_distance_base = np.linalg.norm(radial_xy)
        if radial_distance_base > 1e-6:
            radial_direction = radial_xy / radial_distance_base
        else:
            # Fallback for legs at origin
            if leg_name.startswith('FL'):
                radial_direction = np.array([1.0, 1.0]) / np.sqrt(2)
            elif leg_name.startswith('FR'):
                radial_direction = np.array([1.0, -1.0]) / np.sqrt(2)
            elif leg_name.startswith('RL'):
                radial_direction = np.array([-1.0, 1.0]) / np.sqrt(2)
            else:  # RR
                radial_direction = np.array([-1.0, -1.0]) / np.sqrt(2)
        
        # Compute extension factor based on phase
        if phase < self.phase_crouch_end:
            # Initial crouch: legs compressed, feet pulled inward
            extension_factor = -0.3  # Negative = compression
        elif phase < self.phase_peak_start:
            # Rising phase: smooth extension from compressed to full
            rise_phase = (phase - self.phase_rise_start) / (self.phase_peak_start - self.phase_rise_start)
            # Smooth S-curve extension
            extension_factor = -0.3 + 1.3 * (1 - np.cos(rise_phase * np.pi)) / 2
        else:
            # Peak hold: full extension
            extension_factor = 1.0
        
        # Apply radial displacement
        radial_displacement = self.max_radial_extension * extension_factor
        foot_xy = base_foot[:2] + radial_direction * radial_displacement
        
        # Compute vertical displacement (feet move slightly upward during extension)
        if phase < self.phase_crouch_end:
            # Crouch: feet at lower position
            foot_z = base_foot[2] - 0.05
        elif phase < self.phase_peak_start:
            # Rising: feet gradually move upward
            rise_phase = (phase - self.phase_rise_start) / (self.phase_peak_start - self.phase_rise_start)
            upward_displacement = self.max_upward_foot_displacement * (1 - np.cos(rise_phase * np.pi)) / 2
            foot_z = base_foot[2] - 0.05 + upward_displacement
        else:
            # Peak: feet at elevated position
            foot_z = base_foot[2] - 0.05 + self.max_upward_foot_displacement
        
        return np.array([foot_xy[0], foot_xy[1], foot_z])