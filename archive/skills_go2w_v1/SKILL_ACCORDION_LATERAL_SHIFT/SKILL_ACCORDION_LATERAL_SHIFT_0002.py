from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ACCORDION_LATERAL_SHIFT_MotionGenerator(BaseMotionGenerator):
    """
    Accordion lateral shift gait: rhythmic compression-extension cycle with lateral sliding.
    
    Phase cycle:
    - [0.0-0.3]: All legs compress inward toward centerline (stance)
    - [0.3-0.5]: Compressed stance slides laterally leftward via base velocity
    - [0.5-0.8]: All legs extend outward to widen stance (stance)
    - [0.8-1.0]: Extended stance held for stabilization (stance)
    
    All four feet maintain continuous ground contact throughout.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.6  # Cycle frequency (Hz)
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Lateral compression/extension parameters
        self.compression_amplitude = 0.08  # Meters to compress inward from base position
        self.extension_amplitude = 0.06    # Meters to extend outward from base position
        
        # Lateral slide velocity during compressed phase
        self.lateral_slide_velocity = -0.3  # Negative y = leftward (m/s)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Phase boundaries for state machine
        self.phase_compress_start = 0.0
        self.phase_compress_end = 0.3
        self.phase_slide_start = 0.3
        self.phase_slide_end = 0.5
        self.phase_extend_start = 0.5
        self.phase_extend_end = 0.8
        self.phase_hold_start = 0.8
        self.phase_hold_end = 1.0

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on current phase.
        
        - [0.0-0.3]: Zero velocity (compression)
        - [0.3-0.5]: Lateral leftward velocity (slide)
        - [0.5-0.8]: Zero velocity (extension)
        - [0.8-1.0]: Zero velocity (hold)
        """
        
        if self.phase_slide_start <= phase < self.phase_slide_end:
            # Compressed slide phase: lateral velocity
            vy = self.lateral_slide_velocity
        else:
            # All other phases: stationary base
            vy = 0.0
        
        self.vel_world = np.array([0.0, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase.
        
        Lateral (y-axis) modulation:
        - FL and RL: positive y (left side)
        - FR and RR: negative y (right side)
        
        Phase-driven lateral offset:
        - [0.0-0.3]: Compress inward (reduce |y|)
        - [0.3-0.5]: Hold compressed
        - [0.5-0.8]: Extend outward (increase |y|)
        - [0.8-1.0]: Hold extended
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine lateral direction sign based on leg side
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            # Left side: positive y
            lateral_sign = 1.0
        else:
            # Right side (FR, RR): negative y
            lateral_sign = -1.0
        
        # Compute lateral offset based on phase
        if phase < self.phase_compress_end:
            # Compression phase [0.0-0.3]: smoothly compress inward
            progress = phase / self.phase_compress_end
            # Use smooth cosine interpolation
            compression_factor = 0.5 * (1.0 - np.cos(np.pi * progress))
            lateral_offset = -lateral_sign * self.compression_amplitude * compression_factor
            
        elif phase < self.phase_slide_end:
            # Compressed slide phase [0.3-0.5]: hold compressed position
            lateral_offset = -lateral_sign * self.compression_amplitude
            
        elif phase < self.phase_extend_end:
            # Extension phase [0.5-0.8]: smoothly extend outward
            progress = (phase - self.phase_extend_start) / (self.phase_extend_end - self.phase_extend_start)
            extension_factor = 0.5 * (1.0 - np.cos(np.pi * progress))
            # Transition from compressed to extended
            lateral_offset = -lateral_sign * self.compression_amplitude + \
                           lateral_sign * (self.compression_amplitude + self.extension_amplitude) * extension_factor
            
        else:
            # Hold extended phase [0.8-1.0]: hold extended position
            lateral_offset = lateral_sign * self.extension_amplitude
        
        # Apply lateral offset to y-coordinate
        foot_pos = base_pos.copy()
        foot_pos[1] += lateral_offset
        
        # X and Z coordinates remain constant (level stance, no lifting)
        return foot_pos