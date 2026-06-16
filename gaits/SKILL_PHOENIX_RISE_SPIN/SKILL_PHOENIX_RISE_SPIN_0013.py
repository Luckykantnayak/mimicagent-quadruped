from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PHOENIX_RISE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Phoenix Rise Spin: A dramatic rising motion from crouch to maximum height
    while executing a full 360-degree yaw rotation with radially symmetric
    leg extension creating a 'wings spreading' aesthetic.
    
    All four feet remain in ground contact throughout the motion.
    The visual rise is achieved by feet moving downward in body frame while
    the body appears to rise relative to the ground.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.4  # Slow dramatic motion, ~2.5 seconds per cycle
        
        # Base foot positions (nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - reduced to stay within kinematic workspace
        self.max_height_gain = 0.16  # Reduced from 0.28m to avoid joint limits
        self.max_radial_extension = 0.16  # Reduced from 0.22m to avoid overextension
        
        # Total 3D displacement: sqrt(0.16^2 + 0.16^2) = 0.226m, safely within workspace
        
        self.total_yaw_rotation = 2 * np.pi  # 360 degrees in radians
        
        # Phase boundaries for motion stages
        self.phase_crouch_end = 0.2
        self.phase_rise_start = 0.2
        self.phase_mid_rise = 0.5
        self.phase_peak_start = 0.8
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(self, root_pos, root_quat):
        """Reset base state"""
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0

    def smooth_transition(self, phase, start, end, power=1.5):
        """Smooth transition function with adjustable power for easing"""
        if phase < start:
            return 0.0
        elif phase > end:
            return 1.0
        else:
            t = (phase - start) / (end - start)
            # Smooth S-curve using cosine-based easing for better smoothness
            return (1.0 - np.cos(t * np.pi)) / 2.0

    def update_base_motion(self, phase, dt):
        """
        Update base motion with yaw rotation only.
        No vertical velocity - the rise effect is achieved through foot motion.
        
        Phase 0.0-0.2: Stationary crouch
        Phase 0.2-1.0: Continuous yaw rotation with smooth acceleration/deceleration
        """
        
        # No vertical velocity - maintain ground contact
        vz = 0.0
        
        # Compute yaw rate with smooth acceleration and deceleration
        if phase < self.phase_rise_start:
            # No rotation during initial crouch
            yaw_rate = 0.0
        else:
            # Smooth rotation from phase 0.2 to 1.0
            rotation_phase = (phase - self.phase_rise_start) / (1.0 - self.phase_rise_start)
            
            # Use sine-based velocity profile for smooth acceleration and deceleration
            # This creates smooth start and end to rotation
            rotation_duration = (1.0 - self.phase_rise_start) / self.freq
            avg_yaw_rate = self.total_yaw_rotation / rotation_duration
            
            # Smooth bell curve profile using sine
            velocity_profile = np.sin(rotation_phase * np.pi)
            yaw_rate = avg_yaw_rate * velocity_profile * 1.5  # Scale to compensate for sine average
        
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
        Compute foot position in body frame with radial extension and downward motion.
        
        All legs extend radially outward as feet move downward in body frame,
        creating the visual effect of the body rising while maintaining ground contact.
        Motion is decoupled: radial extension leads, vertical descent follows.
        """
        
        # Get base foot position
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial direction from body center (in XY plane)
        radial_xy = np.array([base_foot[0], base_foot[1]])
        radial_distance_base = np.linalg.norm(radial_xy)
        if radial_distance_base > 1e-6:
            radial_direction = radial_xy / radial_distance_base
        else:
            # Fallback for legs near origin - assign direction based on leg name
            if 'FL' in leg_name:
                radial_direction = np.array([1.0, 1.0]) / np.sqrt(2)
            elif 'FR' in leg_name:
                radial_direction = np.array([1.0, -1.0]) / np.sqrt(2)
            elif 'RL' in leg_name:
                radial_direction = np.array([-1.0, 1.0]) / np.sqrt(2)
            else:  # RR
                radial_direction = np.array([-1.0, -1.0]) / np.sqrt(2)
        
        # Compute extension factor with phased motion (radial extension leads)
        if phase < self.phase_crouch_end:
            # Initial crouch: legs compressed, feet pulled inward moderately
            extension_factor = -0.30  # Moderate compression
        elif phase < self.phase_mid_rise:
            # Early rise: rapid radial extension (most radial motion happens here)
            rise_progress = self.smooth_transition(phase, self.phase_rise_start, self.phase_mid_rise, power=1.3)
            extension_factor = -0.30 + 1.10 * rise_progress  # Reaches 0.80 at mid_rise
        elif phase < self.phase_peak_start:
            # Late rise: complete remaining radial extension
            rise_progress = self.smooth_transition(phase, self.phase_mid_rise, self.phase_peak_start, power=1.5)
            extension_factor = 0.80 + 0.20 * rise_progress  # Completes to 1.0
        else:
            # Peak hold: full extension
            extension_factor = 1.0
        
        # Apply radial displacement
        radial_displacement = self.max_radial_extension * extension_factor
        foot_xy = base_foot[:2] + radial_direction * radial_displacement
        
        # Compute vertical displacement with phased motion (vertical descent follows radial)
        if phase < self.phase_crouch_end:
            # Crouch: feet at nominal height
            vertical_offset = 0.0
        elif phase < self.phase_mid_rise:
            # Early rise: minimal vertical descent (20% of total)
            rise_progress = self.smooth_transition(phase, self.phase_rise_start, self.phase_mid_rise, power=1.2)
            vertical_offset = -self.max_height_gain * 0.20 * rise_progress
        elif phase < self.phase_peak_start:
            # Late rise: major vertical descent (remaining 80%)
            rise_progress = self.smooth_transition(phase, self.phase_mid_rise, self.phase_peak_start, power=1.4)
            vertical_offset = -self.max_height_gain * (0.20 + 0.80 * rise_progress)
        else:
            # Peak: feet at maximum downward position
            vertical_offset = -self.max_height_gain
        
        # Apply vertical offset to maintain ground contact
        foot_z = base_foot[2] + vertical_offset
        
        return np.array([foot_xy[0], foot_xy[1], foot_z])