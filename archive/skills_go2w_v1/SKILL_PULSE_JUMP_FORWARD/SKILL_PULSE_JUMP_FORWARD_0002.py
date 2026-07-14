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
        self.small_amplitude = 0.06
        self.medium_amplitude = 0.085
        self.large_amplitude = 0.10
        
        # Crouch depth (scales with amplitude)
        self.small_crouch_depth = 0.03
        self.medium_crouch_depth = 0.045
        self.large_crouch_depth = 0.06
        
        # Forward velocity (constant)
        self.forward_velocity = 0.5
        
        # Compute initial base height to ensure feet start on ground
        avg_foot_z = np.mean([pos[2] for pos in initial_foot_positions_body.values()])
        self.nominal_base_height = -avg_foot_z + 0.01
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.nominal_base_height])
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
        return self.jump_phases[-1]

    def _compute_jump_sub_phase(self, phase, jump_start, jump_end):
        """
        Convert global phase to local jump phase [0,1].
        Each jump has internal structure:
        - [0.0, 0.30]: crouch (grounded)
        - [0.30, 0.40]: launch/extension (grounded)
        - [0.40, 0.70]: flight (aerial)
        - [0.70, 1.00]: landing/absorption (grounded)
        """
        jump_duration = jump_end - jump_start
        local_phase = (phase - jump_start) / jump_duration
        return np.clip(local_phase, 0.0, 1.0)

    def _get_base_height_offset(self, local_phase, jump_params):
        """
        Compute desired base height offset from nominal based on jump phase.
        Negative offset = crouch down, positive offset = jump up.
        """
        height_offset = 0.0
        
        # Crouch phase (0.0-0.30): lower base while feet stay on ground
        if local_phase < 0.30:
            crouch_progress = local_phase / 0.30
            # Smooth crouch using cosine for continuity
            crouch_blend = 0.5 - 0.5 * np.cos(crouch_progress * np.pi)
            height_offset = -jump_params["crouch"] * crouch_blend
        
        # Launch phase (0.30-0.40): explosive extension from crouch to liftoff
        elif 0.30 <= local_phase < 0.40:
            launch_progress = (local_phase - 0.30) / 0.10
            # Smooth transition from crouch to flight start
            launch_blend = 0.5 - 0.5 * np.cos(launch_progress * np.pi)
            height_offset = -jump_params["crouch"] * (1.0 - launch_blend) + jump_params["amp"] * 0.1 * launch_blend
        
        # Flight phase (0.40-0.70): parabolic trajectory
        elif 0.40 <= local_phase < 0.70:
            flight_progress = (local_phase - 0.40) / 0.30
            # Parabolic: rise to peak at midpoint, then descend
            parabola = 1.0 - 4.0 * (flight_progress - 0.5) ** 2
            height_offset = jump_params["amp"] * parabola
        
        # Landing phase (0.70-1.00): absorb impact and return to neutral
        elif 0.70 <= local_phase <= 1.0:
            landing_progress = (local_phase - 0.70) / 0.30
            # Smooth landing absorption with damped return
            if landing_progress < 0.5:
                # Initial impact absorption
                absorption_blend = landing_progress * 2.0
                compression_amount = 0.5 * jump_params["crouch"] * (0.5 - 0.5 * np.cos(absorption_blend * np.pi))
                height_offset = -compression_amount
            else:
                # Recovery to neutral
                recovery_blend = (landing_progress - 0.5) * 2.0
                recovery_amount = 0.5 * jump_params["crouch"] * (1.0 - (0.5 - 0.5 * np.cos(recovery_blend * np.pi)))
                height_offset = -recovery_amount
        
        return height_offset

    def update_base_motion(self, phase, dt):
        """
        Update base pose with constant forward velocity and vertical motion based on jump phase.
        Base height varies smoothly through crouch-launch-flight-landing cycle.
        """
        jump_params = self._get_current_jump_params(phase)
        local_phase = self._compute_jump_sub_phase(phase, jump_params["start"], jump_params["end"])
        
        # Compute target base height
        height_offset = self._get_base_height_offset(local_phase, jump_params)
        target_base_height = self.nominal_base_height + height_offset
        
        # Compute vertical velocity to reach target height
        current_height = self.root_pos[2]
        height_error = target_base_height - current_height
        
        # Use proportional control with damping for smooth tracking
        vz = height_error * 15.0
        vz = np.clip(vz, -3.0, 3.0)
        
        # Constant forward velocity
        vx = self.forward_velocity
        
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
        During ground contact phases, feet compensate for base height changes to maintain ground contact.
        During flight phase, feet lift relative to body.
        """
        jump_params = self._get_current_jump_params(phase)
        local_phase = self._compute_jump_sub_phase(phase, jump_params["start"], jump_params["end"])
        
        # Start from nominal foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if in flight or ground contact phase
        in_flight = 0.40 <= local_phase < 0.70
        
        if in_flight:
            # During flight: lift feet slightly relative to body for ground clearance
            flight_progress = (local_phase - 0.40) / 0.30
            # Smooth lift and lower using parabolic profile
            lift_amount = 0.02 * (1.0 - 4.0 * (flight_progress - 0.5) ** 2)
            foot[2] += lift_amount
        else:
            # During ground contact: compensate for base height changes
            # Feet need to move up in body frame when base goes down (crouch)
            # and move down in body frame when base goes up (extension)
            height_offset = self._get_base_height_offset(local_phase, jump_params)
            
            # Compensate in opposite direction to maintain ground contact
            foot[2] -= height_offset
        
        return foot