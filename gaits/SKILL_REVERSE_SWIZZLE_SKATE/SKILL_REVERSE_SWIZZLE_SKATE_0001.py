from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERSE_SWIZZLE_SKATE_MotionGenerator(BaseMotionGenerator):
    """
    Reverse swizzle skating motion with synchronized leg swizzling.
    
    - All four legs move in perfect synchronization
    - Legs swizzle outward and inward in smooth arcs
    - Continuous ground contact throughout entire cycle
    - Base moves backward with velocity modulated by swizzle phase
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower cycle for smooth swizzling motion
        
        # Base foot positions (centerline configuration)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Swizzle parameters
        self.lateral_amplitude = 0.15  # Maximum lateral displacement from centerline
        self.longitudinal_shift = 0.05  # Slight forward shift during outward push
        
        # Backward velocity parameters (modulated by phase)
        self.vx_min = -0.3  # Minimum backward velocity (centerline phases)
        self.vx_max = -0.8  # Maximum backward velocity (extension phase)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # All legs synchronized (zero phase offset)
        self.phase_offsets = {
            leg: 0.0 for leg in leg_names
        }

    def update_base_motion(self, phase, dt):
        """
        Update base with backward velocity modulated by swizzle cycle.
        
        Velocity profile:
        - phase [0.0, 0.3]: moderate backward drift (centerline start)
        - phase [0.3, 0.6]: increasing backward velocity (outward push)
        - phase [0.6, 0.8]: peak backward velocity (maximum extension)
        - phase [0.8, 1.0]: decreasing backward velocity (inward recovery)
        """
        
        # Compute phase-dependent backward velocity using smooth interpolation
        if phase < 0.3:
            # Centerline start: moderate velocity
            progress = phase / 0.3
            vx = self.vx_min + (self.vx_max - self.vx_min) * 0.3 * (1 - np.cos(np.pi * progress)) / 2
        elif phase < 0.6:
            # Outward push: increasing to peak velocity
            progress = (phase - 0.3) / 0.3
            vx = self.vx_min + (self.vx_max - self.vx_min) * (0.3 + 0.6 * progress)
        elif phase < 0.8:
            # Maximum extension: maintain peak velocity
            vx = self.vx_max
        else:
            # Inward recovery: decreasing velocity
            progress = (phase - 0.8) / 0.2
            vx = self.vx_max - (self.vx_max - self.vx_min) * 0.6 * progress
        
        # Set velocity commands (world frame)
        self.vel_world = np.array([vx, 0.0, 0.0])
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
        Compute foot position for synchronized swizzle motion.
        
        All legs move together through:
        1. Centerline start (phase 0.0-0.3): minimal lateral offset
        2. Outward push (phase 0.3-0.6): smooth arc outward
        3. Maximum extension (phase 0.6-0.8): hold at peak lateral position
        4. Inward recovery (phase 0.8-1.0): smooth arc back to centerline
        """
        
        # Get base foot position
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine lateral direction (left legs negative, right legs positive)
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            lateral_sign = -1.0  # Left side
        else:
            lateral_sign = 1.0  # Right side
        
        # Determine longitudinal direction (front legs positive, rear legs negative)
        if leg_name.startswith('FL') or leg_name.startswith('FR'):
            longitudinal_sign = 1.0  # Front legs
        else:
            longitudinal_sign = -1.0  # Rear legs
        
        # Compute smooth swizzle trajectory using continuous function
        if phase < 0.3:
            # Centerline start: minimal lateral spread
            progress = phase / 0.3
            lateral_offset = self.lateral_amplitude * 0.1 * np.sin(np.pi * progress / 2)
            longitudinal_offset = 0.0
        elif phase < 0.6:
            # Outward push: smooth arc trajectory
            progress = (phase - 0.3) / 0.3
            # Sinusoidal outward motion for smooth acceleration/deceleration
            lateral_offset = self.lateral_amplitude * (0.1 + 0.9 * (1 - np.cos(np.pi * progress)) / 2)
            # Slight forward shift during push
            longitudinal_offset = self.longitudinal_shift * np.sin(np.pi * progress)
        elif phase < 0.8:
            # Maximum extension: hold position
            lateral_offset = self.lateral_amplitude
            longitudinal_offset = 0.0
        else:
            # Inward recovery: smooth arc back to centerline
            progress = (phase - 0.8) / 0.2
            # Sinusoidal inward motion
            lateral_offset = self.lateral_amplitude * (1 - progress) * (1 + np.cos(np.pi * progress)) / 2
            longitudinal_offset = 0.0
        
        # Apply offsets to base foot position
        foot = base_foot.copy()
        foot[0] += longitudinal_sign * longitudinal_offset  # x: forward/backward
        foot[1] += lateral_sign * lateral_offset  # y: lateral (left/right)
        foot[2] = base_foot[2]  # z: maintain ground contact (no vertical motion)
        
        return foot