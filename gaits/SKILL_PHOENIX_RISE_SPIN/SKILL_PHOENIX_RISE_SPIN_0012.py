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
        
        # Motion parameters
        self.max_height_gain = 0.28  # Visual rise effect in meters
        self.total_yaw_rotation = 2 * np.pi  # 360 degrees in radians
        self.max_radial_extension = 0.22  # Maximum outward foot displacement
        
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

    def update_base_motion(self, phase, dt):
        """
        Update base motion with yaw rotation only.
        No vertical velocity - the rise effect is achieved through foot motion.
        
        Phase 0.0-0.2: Stationary crouch
        Phase 0.2-1.0: Continuous yaw rotation
        """
        
        # No vertical velocity - maintain ground contact
        vz = 0.0
        
        # Compute yaw rate with smooth acceleration and deceleration
        if phase < self.phase_rise_start:
            # No rotation during initial crouch
            yaw_rate = 0.0
        elif phase < self.phase_peak_start:
            # Smooth rotation during main rise phase
            rotation_phase = (phase - self.phase_rise_start) / (self.phase_peak_start - self.phase_rise_start)
            # Use smoothed profile for yaw rate
            smooth_factor = (1 - np.cos(rotation_phase * np.pi)) / 2
            rotation_duration = (self.phase_peak_start - self.phase_rise_start) / self.freq
            # Target most of rotation during main phase
            target_rotation = self.total_yaw_rotation * 0.85
            avg_yaw_rate = target_rotation / rotation_duration
            yaw_rate = avg_yaw_rate * (1.0 + 0.3 * np.sin(rotation_phase * np.pi))
        else:
            # Complete final rotation during hold phase
            final_phase = (phase - self.phase_peak_start) / (1.0 - self.phase_peak_start)
            rotation_duration = (1.0 - self.phase_peak_start) / self.freq
            remaining_rotation = self.total_yaw_rotation * 0.15
            avg_yaw_rate = remaining_rotation / rotation_duration
            yaw_rate = avg_yaw_rate * (1.0 - final_phase)
        
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

    def smooth_transition(self, phase, start, end, power=2.0):
        """Smooth transition function with adjustable power for easing"""
        if phase < start:
            return 0.0
        elif phase > end:
            return 1.0
        else:
            t = (phase - start) / (end - start)
            # Smooth S-curve using polynomial easing
            return t ** power * (3.0 - 2.0 * t)

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with radial extension and downward motion.
        
        All legs extend radially outward as feet move downward in body frame,
        creating the visual effect of the body rising while maintaining ground contact.
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
        
        # Compute extension factor based on phase with smooth transitions
        if phase < self.phase_crouch_end:
            # Initial crouch: legs compressed, feet pulled inward
            extension_factor = -0.25  # Moderate compression
        elif phase < self.phase_peak_start:
            # Rising phase: smooth extension from compressed to full
            rise_progress = self.smooth_transition(phase, self.phase_rise_start, self.phase_peak_start, power=1.5)
            extension_factor = -0.25 + 1.25 * rise_progress
        else:
            # Peak hold: full extension
            extension_factor = 1.0
        
        # Apply radial displacement with smooth transitions
        radial_displacement = self.max_radial_extension * extension_factor
        foot_xy = base_foot[:2] + radial_direction * radial_displacement
        
        # Compute vertical displacement: feet move DOWN in body frame to maintain ground contact
        # This creates the visual effect of the body rising
        if phase < self.phase_crouch_end:
            # Crouch: feet at nominal height (no downward offset yet)
            vertical_offset = 0.0
        elif phase < self.phase_peak_start:
            # Rising phase: feet gradually move downward in body frame
            rise_progress = self.smooth_transition(phase, self.phase_rise_start, self.phase_peak_start, power=1.8)
            vertical_offset = -self.max_height_gain * rise_progress
        else:
            # Peak: feet at maximum downward position in body frame
            vertical_offset = -self.max_height_gain
        
        # Apply vertical offset to maintain ground contact
        foot_z = base_foot[2] + vertical_offset
        
        return np.array([foot_xy[0], foot_xy[1], foot_z])