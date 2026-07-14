from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PULSE_JUMP_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Pulse Jump Forward skill.
    
    The robot performs continuous synchronized four-legged jumps with varying amplitude:
    - Small jump (phase 0.0-0.2)
    - Medium jump (phase 0.2-0.4)
    - Large jump (phase 0.4-0.6)
    - Medium jump (phase 0.6-0.8)
    - Small jump (phase 0.8-1.0)
    
    All four legs act in perfect synchronization (crouch, launch, flight, land together).
    Forward velocity is constant throughout. Vertical motion varies with amplitude envelope.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Jump parameters
        self.small_amplitude = 0.06  # 60% of peak
        self.medium_amplitude = 0.085  # 85% of peak
        self.large_amplitude = 0.10  # 100% peak
        
        # Crouch depth (scales with amplitude)
        self.small_crouch_depth = 0.03
        self.medium_crouch_depth = 0.045
        self.large_crouch_depth = 0.06
        
        # Forward velocity (constant)
        self.forward_velocity = 0.5
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Jump phase boundaries
        self.jump_phases = [
            {"start": 0.0, "end": 0.2, "amp": self.small_amplitude, "crouch": self.small_crouch_depth},
            {"start": 0.2, "end": 0.4, "amp": self.medium_amplitude, "crouch": self.medium_crouch_depth},
            {"start": 0.4, "end": 0.6, "amp": self.large_amplitude, "crouch": self.large_crouch_depth},
            {"start": 0.6, "end": 0.8, "amp": self.medium_amplitude, "crouch": self.medium_crouch_depth},
            {"start": 0.8, "end": 1.0, "amp": self.small_amplitude, "crouch": self.small_crouch_depth},
        ]

    def _get_current_jump_params(self, phase):
        """Determine which jump we're in and return its parameters."""
        for jump in self.jump_phases:
            if jump["start"] <= phase < jump["end"]:
                return jump
        # Handle phase = 1.0 edge case
        return self.jump_phases[-1]

    def _compute_jump_sub_phase(self, phase, jump_start, jump_end):
        """
        Convert global phase to local jump phase [0,1].
        Each jump has internal structure:
        - [0.0, 0.25]: crouch
        - [0.25, 0.40]: launch/extension
        - [0.40, 0.75]: flight
        - [0.75, 1.00]: landing/absorption
        """
        jump_duration = jump_end - jump_start
        local_phase = (phase - jump_start) / jump_duration
        return np.clip(local_phase, 0.0, 1.0)

    def update_base_motion(self, phase, dt):
        """
        Update base pose with constant forward velocity and vertical motion based on jump phase.
        """
        jump_params = self._get_current_jump_params(phase)
        local_phase = self._compute_jump_sub_phase(phase, jump_params["start"], jump_params["end"])
        
        # Constant forward velocity
        vx = self.forward_velocity
        
        # Vertical velocity varies with jump phase
        vz = 0.0
        
        # Launch phase (0.25-0.40): initiate upward velocity
        if 0.25 <= local_phase < 0.40:
            launch_progress = (local_phase - 0.25) / 0.15
            # Peak vertical velocity scales with amplitude
            peak_vz = jump_params["amp"] * 8.0
            vz = peak_vz * launch_progress
        
        # Flight phase (0.40-0.75): parabolic trajectory
        elif 0.40 <= local_phase < 0.75:
            flight_progress = (local_phase - 0.40) / 0.35
            peak_vz = jump_params["amp"] * 8.0
            # Parabolic: starts positive, goes to zero at apex, then negative
            vz = peak_vz * (1.0 - 2.0 * flight_progress)
        
        # Landing phase (0.75-1.00): downward velocity damping to zero
        elif 0.75 <= local_phase <= 1.0:
            landing_progress = (local_phase - 0.75) / 0.25
            peak_vz = jump_params["amp"] * 8.0
            # Decelerate from negative to zero
            vz = -peak_vz * (1.0 - landing_progress)
        
        # Set velocity commands
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.zeros(3)
        
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
        Compute foot position in body frame for synchronized jumping motion.
        All legs follow identical vertical motion pattern scaled by jump amplitude.
        """
        jump_params = self._get_current_jump_params(phase)
        local_phase = self._compute_jump_sub_phase(phase, jump_params["start"], jump_params["end"])
        
        # Start from nominal foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Vertical displacement based on jump sub-phase
        z_offset = 0.0
        
        # Crouch phase (0.0-0.25): compress legs downward
        if local_phase < 0.25:
            crouch_progress = local_phase / 0.25
            # Smooth crouch: ease in and out
            crouch_amount = np.sin(crouch_progress * np.pi / 2)
            z_offset = -jump_params["crouch"] * crouch_amount
        
        # Launch phase (0.25-0.40): rapid extension from crouch to neutral
        elif 0.25 <= local_phase < 0.40:
            launch_progress = (local_phase - 0.25) / 0.15
            # Extend from crouch back to neutral
            z_offset = -jump_params["crouch"] * (1.0 - launch_progress)
        
        # Flight phase (0.40-0.75): feet maintain extended position (body moves relative to world)
        elif 0.40 <= local_phase < 0.75:
            z_offset = 0.0
        
        # Landing phase (0.75-1.00): absorb impact by compressing then returning to neutral
        elif 0.75 <= local_phase <= 1.0:
            landing_progress = (local_phase - 0.75) / 0.25
            # Compress on impact, then return to neutral
            # Use parabolic shape: compress quickly, extend slowly
            if landing_progress < 0.5:
                absorption_amount = np.sin(landing_progress * 2.0 * np.pi / 2)
                z_offset = -jump_params["crouch"] * 0.7 * absorption_amount
            else:
                recovery_progress = (landing_progress - 0.5) * 2.0
                z_offset = -jump_params["crouch"] * 0.7 * (1.0 - recovery_progress)
        
        foot[2] += z_offset
        
        return foot