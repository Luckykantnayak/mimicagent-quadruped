from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ACCORDION_LATERAL_SHIFT_MotionGenerator(BaseMotionGenerator):
    """
    Accordion lateral shift locomotion.
    
    The robot performs sideways locomotion by rhythmically compressing its body width
    (tucking legs inward), sliding laterally while compressed, then extending its width
    (spreading legs outward). This accordion-like motion produces net lateral displacement.
    
    Phase structure:
    - [0.0, 0.3]: Lateral compression - legs tuck inward
    - [0.3, 0.5]: Compressed lateral slide - base moves leftward (positive y)
    - [0.5, 0.8]: Lateral extension - legs extend outward
    - [0.8, 1.0]: Anchored preparation - legs hold extended stance
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        # Initialize base class
        BaseMotionGenerator.__init__(self, initial_foot_positions_body, freq=1.0)
        
        self.leg_names = leg_names
        self.freq = 0.5  # Cycle frequency (Hz)
        
        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Accordion motion parameters (reduced from 0.08 to avoid joint limits)
        self.compression_amount = 0.04  # How much to tuck legs inward (m)
        self.slide_velocity = 0.3  # Lateral velocity during compressed slide (m/s)
        
        # Phase transition points
        self.phase_compression_end = 0.3
        self.phase_slide_end = 0.5
        self.phase_extension_end = 0.8
        
        # Transition blending width for smooth derivatives
        self.blend_width = 0.05
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base motion based on current phase.
        
        Only apply lateral velocity during compressed slide phase [0.3, 0.5].
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        # Smooth transition into and out of slide phase
        slide_start = self.phase_compression_end
        slide_end = self.phase_slide_end
        
        if phase < slide_start - self.blend_width:
            # Before slide
            vy = 0.0
        elif phase < slide_start + self.blend_width:
            # Blend into slide
            blend_t = (phase - (slide_start - self.blend_width)) / (2.0 * self.blend_width)
            blend_t = np.clip(blend_t, 0.0, 1.0)
            vy = self.slide_velocity * self._quintic_step(blend_t)
        elif phase < slide_end - self.blend_width:
            # Full slide
            vy = self.slide_velocity
        elif phase < slide_end + self.blend_width:
            # Blend out of slide
            blend_t = (phase - (slide_end - self.blend_width)) / (2.0 * self.blend_width)
            blend_t = np.clip(blend_t, 0.0, 1.0)
            vy = self.slide_velocity * (1.0 - self._quintic_step(blend_t))
        else:
            # After slide
            vy = 0.0
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame based on phase.
        
        All feet remain in ground contact throughout motion.
        Lateral (y) position modulates according to compression/extension.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is on left or right side
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Get base lateral offset
        base_y = self.base_feet_pos_body[leg_name][1]
        
        # Compute lateral offset modulation based on phase
        lateral_offset = self._compute_lateral_offset(phase)
        
        # Apply lateral offset (inward is toward centerline)
        if is_left_leg:
            # Left legs: move toward centerline means increase y (less negative)
            foot[1] = base_y + lateral_offset
        else:
            # Right legs: move toward centerline means decrease y (less positive)
            foot[1] = base_y - lateral_offset
        
        # Ensure foot maintains ground contact by preserving z coordinate
        # The base_feet_pos_body should already have appropriate z values for ground contact
        foot[2] = self.base_feet_pos_body[leg_name][2]
        
        return foot

    def _compute_lateral_offset(self, phase):
        """
        Compute lateral offset from base position with C2 continuity.
        
        Returns positive offset during compression (legs move toward centerline).
        Returns zero during extended stance.
        Uses smooth quintic interpolation for C2 continuity.
        """
        
        # Define phase boundaries with blending
        compress_start = 0.0
        compress_end = self.phase_compression_end
        slide_end = self.phase_slide_end
        extend_end = self.phase_extension_end
        cycle_end = 1.0
        
        # Phase 1: Lateral compression [0.0, 0.3]
        if phase <= compress_end:
            progress = phase / compress_end
            offset = self.compression_amount * self._quintic_step(progress)
            return offset
        
        # Phase 2: Compressed lateral slide [0.3, 0.5]
        elif phase <= slide_end:
            # Hold compressed position
            return self.compression_amount
        
        # Phase 3: Lateral extension [0.5, 0.8]
        elif phase <= extend_end:
            phase_duration = extend_end - slide_end
            progress = (phase - slide_end) / phase_duration
            offset = self.compression_amount * (1.0 - self._quintic_step(progress))
            return offset
        
        # Phase 4: Anchored preparation [0.8, 1.0]
        else:
            return 0.0

    def _quintic_step(self, t):
        """
        Quintic smooth step function with C2 continuity.
        
        Maps [0,1] -> [0,1] with zero first and second derivatives at endpoints.
        Formula: 6t^5 - 15t^4 + 10t^3
        """
        t = np.clip(t, 0.0, 1.0)
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)