from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_BELLOWS_PUMP_LATERAL_MotionGenerator(BaseMotionGenerator):
    """
    Bellows pump lateral locomotion skill.
    
    Cyclic lateral locomotion where the robot compresses and expands its body width
    like accordion bellows, creating a pumping motion that propels the robot leftward.
    All four legs remain in contact throughout, moving symmetrically inward during
    compression phases and outward during expansion phases.
    
    Phase structure:
    - [0.0, 0.25]: compression_1 - legs draw inward, minimal leftward velocity
    - [0.25, 0.5]: expansion_1 - legs extend outward, strong leftward velocity surge
    - [0.5, 0.75]: compression_2 - legs retract inward, minimal leftward velocity
    - [0.75, 1.0]: expansion_2 - legs extend outward, strong leftward velocity surge
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions (nominal stance in body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Bellows parameters
        self.lateral_compression_amplitude = 0.08  # How much legs move inward (m)
        self.lateral_expansion_amplitude = 0.08    # How much legs move outward (m)
        self.longitudinal_offset = 0.02            # Forward/backward offset during cycle (m)
        
        # Velocity commands
        self.vy_compression = 0.15   # Low leftward velocity during compression (m/s)
        self.vy_expansion = 0.6      # High leftward velocity during expansion (m/s)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Identify leg laterality (left legs have positive y in body frame)
        self.leg_laterality = {}
        for leg_name in self.leg_names:
            base_y = self.base_feet_pos_body[leg_name][1]
            self.leg_laterality[leg_name] = 1.0 if base_y > 0 else -1.0
        
        # Identify leg longitudinality (front legs have positive x in body frame)
        self.leg_longitudinality = {}
        for leg_name in self.leg_names:
            base_x = self.base_feet_pos_body[leg_name][0]
            self.leg_longitudinality[leg_name] = 1.0 if base_x > 0 else -1.0

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        Compression phases (0.0-0.25, 0.5-0.75): low leftward velocity
        Expansion phases (0.25-0.5, 0.75-1.0): high leftward velocity surge
        """
        
        # Determine if in compression or expansion phase
        if (0.0 <= phase < 0.25) or (0.5 <= phase < 0.75):
            # Compression phases
            vy = self.vy_compression
        else:
            # Expansion phases (0.25-0.5 or 0.75-1.0)
            vy = self.vy_expansion
        
        # Set velocity commands (leftward is positive y in world frame)
        self.vel_world = np.array([0.0, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
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
        Compute foot position in body frame for given leg and phase.
        
        All feet remain in contact. Feet move laterally in/out to create bellows effect:
        - Compression: feet move inward toward body centerline
        - Expansion: feet move outward away from centerline
        
        Slight forward/backward motion maintains balance during width changes.
        """
        
        # Start from base position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Get leg laterality and longitudinality
        lateral_sign = self.leg_laterality[leg_name]  # +1 for left, -1 for right
        longitudinal_sign = self.leg_longitudinality[leg_name]  # +1 for front, -1 for rear
        
        # Compute lateral offset based on phase (bellows compression/expansion)
        # Compression phases: negative offset (move inward)
        # Expansion phases: positive offset (move outward)
        
        if 0.0 <= phase < 0.25:
            # Compression 1: smooth transition from neutral to compressed
            local_progress = phase / 0.25
            lateral_offset = -self.lateral_compression_amplitude * self._smooth_step(local_progress)
            longitudinal_offset = self.longitudinal_offset * self._smooth_step(local_progress)
            
        elif 0.25 <= phase < 0.5:
            # Expansion 1: smooth transition from compressed to expanded
            local_progress = (phase - 0.25) / 0.25
            lateral_offset = -self.lateral_compression_amplitude + \
                            (self.lateral_compression_amplitude + self.lateral_expansion_amplitude) * \
                            self._smooth_step(local_progress)
            longitudinal_offset = self.longitudinal_offset * (1.0 - self._smooth_step(local_progress))
            
        elif 0.5 <= phase < 0.75:
            # Compression 2: smooth transition from expanded to compressed
            local_progress = (phase - 0.5) / 0.25
            lateral_offset = self.lateral_expansion_amplitude * (1.0 - self._smooth_step(local_progress)) + \
                            (-self.lateral_compression_amplitude) * self._smooth_step(local_progress)
            longitudinal_offset = -self.longitudinal_offset * self._smooth_step(local_progress)
            
        else:  # 0.75 <= phase < 1.0
            # Expansion 2: smooth transition from compressed to expanded (back to start)
            local_progress = (phase - 0.75) / 0.25
            lateral_offset = -self.lateral_compression_amplitude + \
                            (self.lateral_compression_amplitude + self.lateral_expansion_amplitude) * \
                            self._smooth_step(local_progress)
            longitudinal_offset = -self.longitudinal_offset * (1.0 - self._smooth_step(local_progress))
        
        # Apply lateral offset (symmetric about sagittal plane)
        foot[1] += lateral_sign * lateral_offset
        
        # Apply longitudinal offset (front legs move forward during compression, rear move back)
        # This maintains proper foot placement and prevents excessive drift
        foot[0] += longitudinal_sign * longitudinal_offset
        
        return foot
    
    def _smooth_step(self, t):
        """
        Smooth step function for continuous transitions.
        Uses smoothstep interpolation: 3t^2 - 2t^3
        """
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)