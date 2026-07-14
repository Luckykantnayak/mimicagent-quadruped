from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_HOURGLASS_EXPANSION_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Hourglass Expansion Walk: Forward walking gait with rhythmic stance width modulation.
    
    - All four legs converge inward (narrow stance) then expand outward (wide stance) synchronously
    - Base height rises during narrow stance, lowers during wide stance
    - Continuous forward velocity throughout cycle (peaks during narrow stance)
    - All feet remain in ground contact (stance-modulation gait, no swing phase)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Cycle frequency (Hz) - slower for smooth hourglass effect
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Stance width modulation parameters
        self.narrow_width_factor = 0.4  # Minimum lateral width (40% of nominal)
        self.wide_width_factor = 1.6    # Maximum lateral width (160% of nominal)
        
        # Forward motion parameters
        self.forward_velocity_min = 0.3   # Minimum forward velocity (m/s)
        self.forward_velocity_max = 0.8   # Maximum forward velocity during narrow stance (m/s)
        
        # Base height modulation parameters
        self.base_height_amplitude = 0.08  # Vertical oscillation amplitude (m)
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def compute_stance_width_factor(self, phase):
        """
        Compute stance width modulation factor as function of phase.
        
        - [0.0-0.4]: Converge inward (factor decreases from mid to minimum)
        - [0.4-0.8]: Expand outward (factor increases from minimum to maximum)
        - [0.8-1.0]: Begin convergence (factor decreases from maximum toward mid)
        
        Returns factor in range [narrow_width_factor, wide_width_factor]
        """
        mid_width = (self.narrow_width_factor + self.wide_width_factor) / 2.0
        
        if phase < 0.4:
            # Convergence phase: interpolate from mid to narrow
            local_progress = phase / 0.4
            # Smooth cosine interpolation
            blend = 0.5 * (1.0 + np.cos(np.pi * (1.0 - local_progress)))
            return mid_width + (self.narrow_width_factor - mid_width) * blend
        elif phase < 0.8:
            # Expansion phase: interpolate from narrow to wide
            local_progress = (phase - 0.4) / 0.4
            # Smooth sinusoidal transition
            blend = 0.5 * (1.0 - np.cos(np.pi * local_progress))
            return self.narrow_width_factor + (self.wide_width_factor - self.narrow_width_factor) * blend
        else:
            # Convergence initiation: interpolate from wide toward mid
            local_progress = (phase - 0.8) / 0.2
            # Smooth transition back toward mid
            blend = 0.5 * (1.0 + np.cos(np.pi * local_progress))
            return self.wide_width_factor + (mid_width - self.wide_width_factor) * blend

    def compute_forward_velocity(self, phase):
        """
        Compute forward velocity (vx) as function of phase.
        
        - Peak velocity during narrow stance [0.2-0.4]
        - Reduced velocity during wide stance [0.4-0.8]
        - Smooth transitions, never zero
        """
        if phase < 0.2:
            # Convergence rise: velocity increasing
            local_progress = phase / 0.2
            return self.forward_velocity_min + (self.forward_velocity_max - self.forward_velocity_min) * local_progress
        elif phase < 0.4:
            # Narrow peak: maximum velocity
            return self.forward_velocity_max
        elif phase < 0.6:
            # Expansion descent: velocity decreasing
            local_progress = (phase - 0.4) / 0.2
            return self.forward_velocity_max - (self.forward_velocity_max - self.forward_velocity_min) * local_progress
        elif phase < 0.8:
            # Wide plateau: minimum velocity (but still moving forward)
            return self.forward_velocity_min
        else:
            # Convergence initiation: velocity increasing toward next cycle
            local_progress = (phase - 0.8) / 0.2
            return self.forward_velocity_min + (self.forward_velocity_max - self.forward_velocity_min) * 0.5 * local_progress

    def compute_vertical_velocity(self, phase):
        """
        Compute vertical velocity (vz) as function of phase.
        
        - Upward during convergence [0.0-0.4, 0.8-1.0]
        - Downward during expansion [0.4-0.8]
        - Inverse correlation with stance width
        """
        if phase < 0.2:
            # Convergence rise: upward velocity increasing
            local_progress = phase / 0.2
            return self.base_height_amplitude * 2.0 * np.sin(np.pi * local_progress)
        elif phase < 0.4:
            # Narrow peak: upward velocity decreasing to zero
            local_progress = (phase - 0.2) / 0.2
            return self.base_height_amplitude * 2.0 * np.sin(np.pi * (1.0 - local_progress))
        elif phase < 0.6:
            # Expansion descent: downward velocity increasing
            local_progress = (phase - 0.4) / 0.2
            return -self.base_height_amplitude * 2.0 * np.sin(np.pi * local_progress)
        elif phase < 0.8:
            # Wide plateau: downward velocity decreasing to zero
            local_progress = (phase - 0.6) / 0.2
            return -self.base_height_amplitude * 2.0 * np.sin(np.pi * (1.0 - local_progress))
        else:
            # Convergence initiation: upward velocity starting
            local_progress = (phase - 0.8) / 0.2
            return self.base_height_amplitude * 2.0 * np.sin(np.pi * local_progress)

    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent velocities.
        
        - Forward velocity modulated with phase (peak during narrow stance)
        - Vertical velocity coordinated inversely with stance width
        - No lateral or angular motion
        """
        vx = self.compute_forward_velocity(phase)
        vz = self.compute_vertical_velocity(phase)
        
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.zeros(3)  # No rotation
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with synchronized lateral modulation.
        
        All legs modulate their lateral positions (body-y) synchronously:
        - Converge inward during [0.0-0.4]
        - Expand outward during [0.4-0.8]
        - Begin convergence during [0.8-1.0]
        
        Symmetry: FL/RL mirror FR/RR across sagittal plane
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Get stance width modulation factor
        width_factor = self.compute_stance_width_factor(phase)
        
        # Modulate lateral position (body-y) symmetrically
        # Left legs (FL, RL): positive body-y
        # Right legs (FR, RR): negative body-y
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            # Left side: modulate positive body-y
            base_pos[1] = abs(self.base_feet_pos_body[leg_name][1]) * width_factor
        elif leg_name.startswith('FR') or leg_name.startswith('RR'):
            # Right side: modulate negative body-y
            base_pos[1] = -abs(self.base_feet_pos_body[leg_name][1]) * width_factor
        
        # Forward motion in body frame: feet slide backward relative to body as body moves forward
        # Simple sinusoidal fore-aft motion to maintain relative positioning
        forward_offset = 0.05 * np.sin(2.0 * np.pi * phase)
        base_pos[0] = self.base_feet_pos_body[leg_name][0] + forward_offset
        
        # Maintain ground contact: z = 0 in body frame (feet stay on ground)
        base_pos[2] = 0.0
        
        return base_pos