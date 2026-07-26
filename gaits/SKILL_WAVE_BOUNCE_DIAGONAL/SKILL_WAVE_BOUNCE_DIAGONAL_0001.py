from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_WAVE_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal wave bounce locomotion skill.
    
    Sequential vertical bounce propagation along diagonal axis:
    RL (phase 0.0-0.25) → FR (0.25-0.5) → RR (0.5-0.75) → FL (0.75-1.0)
    
    Each leg compresses and extends while maintaining ground contact,
    creating a traveling ripple effect that drives diagonal locomotion.
    
    Base motion: rocking angular velocities + diagonal linear velocity.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Hz, cycle frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Bounce parameters
        self.bounce_amplitude = 0.06  # meters, z-axis compression/extension magnitude
        self.bounce_duration = 0.25  # phase units per bounce
        
        # Diagonal velocity parameters
        self.vx_base = 0.4  # m/s, forward component
        self.vy_amplitude = 0.15  # m/s, lateral oscillation amplitude
        self.vz_amplitude = 0.08  # m/s, vertical oscillation amplitude
        
        # Angular velocity parameters (for rocking motion)
        self.roll_rate_amplitude = 0.5  # rad/s
        self.pitch_rate_amplitude = 0.4  # rad/s
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Bounce timing: (start_phase, end_phase) for each leg
        self.bounce_schedule = {
            'RL': (0.0, 0.25),
            'FR': (0.25, 0.5),
            'RR': (0.5, 0.75),
            'FL': (0.75, 1.0)
        }

    def update_base_motion(self, phase, dt):
        """
        Update base pose using diagonal velocity commands and rocking angular rates.
        
        Linear velocity:
        - vx: constant forward drift
        - vy: sinusoidal lateral oscillation (creates diagonal weaving)
        - vz: vertical oscillation synchronized with bounce wave
        
        Angular velocity:
        - roll: oscillates to tilt left-right, loading bounce legs
        - pitch: oscillates to tilt forward-backward, enhancing wave propagation
        - yaw: zero (pure diagonal translation)
        """
        
        # Linear velocity: diagonal drift with vertical oscillation
        vx = self.vx_base + 0.2 * np.sin(2 * np.pi * phase)  # forward, peaks mid-cycle
        vy = self.vy_amplitude * np.sin(4 * np.pi * phase - np.pi/4)  # lateral oscillation
        vz = self.vz_amplitude * np.sin(4 * np.pi * phase)  # vertical bounce oscillation
        
        self.vel_world = np.array([vx, vy, vz])
        
        # Angular velocity: rocking motion to load each bouncing leg
        # Roll: negative (tilt left) phases 0.0-0.25, 0.75-1.0; positive (tilt right) 0.25-0.75
        roll_rate = self.roll_rate_amplitude * np.sin(2 * np.pi * phase - np.pi/2)
        
        # Pitch: negative (tilt backward) phases 0.0-0.25, 0.5-0.75; positive (forward) 0.25-0.5, 0.75-1.0
        pitch_rate = self.pitch_rate_amplitude * np.sin(4 * np.pi * phase)
        
        yaw_rate = 0.0
        
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        # Integrate pose in world frame
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame.
        
        Each leg performs a vertical bounce (compression then extension) during its
        scheduled phase window while maintaining ground contact. Outside the bounce
        window, the foot maintains nominal stance position with small adjustments
        for base rocking compensation.
        
        Bounce profile: sinusoidal z-displacement during assigned phase range.
        """
        
        # Start with base stance position
        foot_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine which leg we're computing for
        leg_key = None
        for key in self.bounce_schedule.keys():
            if leg_name.startswith(key):
                leg_key = key
                break
        
        if leg_key is None:
            return foot_pos
        
        bounce_start, bounce_end = self.bounce_schedule[leg_key]
        
        # Check if current phase is within this leg's bounce window
        if bounce_start <= phase < bounce_end:
            # Compute local progress within bounce window [0, 1]
            local_progress = (phase - bounce_start) / self.bounce_duration
            
            # Sinusoidal bounce profile: compress (descend) then extend (ascend)
            # First half: descend (negative z), second half: ascend (return to nominal)
            bounce_angle = np.pi * local_progress
            z_offset = -self.bounce_amplitude * np.sin(bounce_angle)
            
            foot_pos[2] += z_offset
            
        else:
            # Outside bounce window: apply small compensation for base rocking
            # to maintain smooth contact during other legs' bounces
            
            # Compute phase-dependent stabilization offset
            stabilization_offset = 0.01 * np.sin(2 * np.pi * phase)
            foot_pos[2] += stabilization_offset
        
        return foot_pos