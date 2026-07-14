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
        
        # Base foot positions in body frame - elevate slightly to accommodate bounce
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            elevated_pos = v.copy()
            elevated_pos[2] += 0.035  # Elevate stance to allow compression to ground
            self.base_feet_pos_body[k] = elevated_pos
        
        # Bounce parameters
        self.bounce_amplitude = 0.035  # meters, z-axis lift magnitude from ground contact
        self.bounce_duration = 0.25  # phase units per bounce
        
        # Diagonal velocity parameters
        self.vx_base = 0.35  # m/s, forward component
        self.vy_amplitude = 0.12  # m/s, lateral oscillation amplitude
        self.vz_amplitude = 0.05  # m/s, vertical oscillation amplitude (reduced)
        
        # Angular velocity parameters (for rocking motion)
        self.roll_rate_amplitude = 0.4  # rad/s
        self.pitch_rate_amplitude = 0.3  # rad/s
        
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
        vx = self.vx_base + 0.15 * np.sin(2 * np.pi * phase)  # forward, peaks mid-cycle
        vy = self.vy_amplitude * np.sin(4 * np.pi * phase - np.pi/4)  # lateral oscillation
        vz = self.vz_amplitude * np.sin(4 * np.pi * phase)  # vertical bounce oscillation
        
        self.vel_world = np.array([vx, vy, vz])
        
        # Angular velocity: rocking motion to load each bouncing leg
        # Smooth sinusoidal profiles for roll and pitch
        roll_rate = self.roll_rate_amplitude * np.sin(2 * np.pi * phase - np.pi/2)
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
        
        Each leg performs a vertical bounce (compression to ground, then extension upward) 
        during its scheduled phase window. The bounce profile ensures feet touch ground 
        at maximum compression (midpoint of bounce window) and lift to elevated stance 
        at the boundaries.
        
        Bounce profile: cosine-based z-displacement ensuring non-negative foot height.
        """
        
        # Start with elevated base stance position
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
            
            # Cosine bounce profile: elevated at boundaries, compressed (ground contact) at midpoint
            # At local_progress=0: cos(0)=1 → z_offset=0 (elevated stance)
            # At local_progress=0.5: cos(pi)=-1 → z_offset=-bounce_amplitude (ground contact)
            # At local_progress=1: cos(2*pi)=1 → z_offset=0 (elevated stance)
            bounce_angle = 2 * np.pi * local_progress
            z_offset = -self.bounce_amplitude * (1.0 - np.cos(bounce_angle)) / 2.0
            
            foot_pos[2] += z_offset
            
        else:
            # Outside bounce window: maintain elevated stance with minimal phase-smoothed adjustment
            # Use smooth transition to avoid discontinuities at bounce boundaries
            stabilization_offset = 0.005 * np.sin(2 * np.pi * phase)
            foot_pos[2] += stabilization_offset
        
        return foot_pos