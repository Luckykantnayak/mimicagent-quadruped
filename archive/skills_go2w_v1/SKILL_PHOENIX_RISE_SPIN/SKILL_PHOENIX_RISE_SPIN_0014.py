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
    spreading laterally (wing-like) and tucking inward longitudinally to
    maintain safe knee angles.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.4  # Slow dramatic motion, ~2.5 seconds per cycle
        
        # Base foot positions (nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - component-based displacement for safe kinematics
        self.max_height_gain = 0.16  # Vertical descent in body frame
        self.max_lateral_spread = 0.16  # Sideways spread (Y-axis) - safe for knees
        self.max_longitudinal_tuck = 0.07  # Inward tuck (X-axis) - prevents hyperextension
        
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

    def smooth_transition(self, phase, start, end):
        """Smooth cosine-based transition function"""
        if phase < start:
            return 0.0
        elif phase > end:
            return 1.0
        else:
            t = (phase - start) / (end - start)
            # Smooth S-curve using cosine
            return (1.0 - np.cos(t * np.pi)) / 2.0

    def update_base_motion(self, phase, dt):
        """
        Update base motion with yaw rotation only.
        No vertical velocity - the rise effect is achieved through foot motion.
        
        Phase 0.0-0.2: Stationary crouch
        Phase 0.2-1.0: Continuous yaw rotation with smooth profile
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
            rotation_duration = (1.0 - self.phase_rise_start) / self.freq
            avg_yaw_rate = self.total_yaw_rotation / rotation_duration
            
            # Smooth bell curve profile
            velocity_profile = np.sin(rotation_phase * np.pi)
            yaw_rate = avg_yaw_rate * velocity_profile * 1.57  # Scale to reach 360 degrees
        
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
        Compute foot position in body frame with component-based displacement.
        
        Lateral spread (Y): outward along body width - creates wing appearance
        Longitudinal (X): inward tuck toward body center - prevents knee hyperextension
        Vertical (Z): downward in body frame - creates visual rise effect
        """
        
        # Get base foot position
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg-specific directional signs
        # Left legs (FL, RL) have positive Y (left side)
        # Right legs (FR, RR) have negative Y (right side)
        is_left = 'L' in leg_name and leg_name[1] == 'L'
        lateral_sign = 1.0 if is_left else -1.0
        
        # Front legs (FL, FR) have positive X (forward)
        # Rear legs (RL, RR) have negative X (backward)
        is_front = leg_name[0] == 'F'
        # For tuck: front legs pull backward (negative), rear legs push forward (positive)
        longitudinal_tuck_sign = -1.0 if is_front else 1.0
        
        # Compute extension progress based on phase
        if phase < self.phase_crouch_end:
            # Initial crouch: moderate compression
            extension_progress = -0.20  # Slight inward pull
        elif phase < self.phase_peak_start:
            # Rising phase: smooth extension from compressed to full spread
            rise_progress = self.smooth_transition(phase, self.phase_rise_start, self.phase_peak_start)
            extension_progress = -0.20 + 1.20 * rise_progress  # Goes from -0.20 to 1.0
        else:
            # Peak hold: full extension
            extension_progress = 1.0
        
        # Compute lateral spread (Y-axis motion) - primary wing-spreading motion
        lateral_offset = lateral_sign * self.max_lateral_spread * extension_progress
        
        # Compute longitudinal tuck (X-axis motion) - inward pull to keep knees safe
        # During crouch, feet are neutral or slightly pulled in
        # During extension, feet tuck inward (opposite to radial expansion)
        if extension_progress < 0:
            # During crouch: minimal longitudinal change
            longitudinal_offset = 0.0
        else:
            # During extension: tuck inward as legs spread laterally
            longitudinal_offset = longitudinal_tuck_sign * self.max_longitudinal_tuck * extension_progress
        
        # Apply XY offsets
        foot_x = base_foot[0] + longitudinal_offset
        foot_y = base_foot[1] + lateral_offset
        
        # Compute vertical displacement (Z-axis motion) - downward for visual rise
        if phase < self.phase_crouch_end:
            # Crouch: feet at nominal height
            vertical_offset = 0.0
        elif phase < self.phase_peak_start:
            # Rising phase: feet descend smoothly in body frame
            rise_progress = self.smooth_transition(phase, self.phase_rise_start, self.phase_peak_start)
            vertical_offset = -self.max_height_gain * rise_progress
        else:
            # Peak: feet at maximum downward position
            vertical_offset = -self.max_height_gain
        
        # Apply vertical offset
        foot_z = base_foot[2] + vertical_offset
        
        return np.array([foot_x, foot_y, foot_z])