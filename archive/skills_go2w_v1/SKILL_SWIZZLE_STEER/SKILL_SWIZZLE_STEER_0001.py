from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SWIZZLE_STEER_MotionGenerator(BaseMotionGenerator):
    """
    Ice-skating-like swizzle locomotion for quadruped.
    
    All four legs remain in continuous ground contact and execute synchronized
    lateral swizzle motions. Two complete swizzle strokes occur per phase cycle:
    - Outward sweep: legs move laterally outward in V-formation while sliding backward (propulsive)
    - Inward recovery: legs converge toward centerline while advancing forward (preparatory)
    
    Base maintains constant forward velocity throughout.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Swizzle motion parameters
        self.swizzle_amplitude = 0.12  # Lateral extent of V-shape (meters)
        self.forward_backward_amplitude = 0.08  # Forward/backward motion amplitude
        
        # Base motion parameters
        self.base_forward_velocity = 0.3  # Steady forward velocity (m/s)
        
        # Store base foot positions (narrow stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Lateral sign for each leg (positive y is left, negative y is right)
        self.lateral_sign = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RL'):
                self.lateral_sign[leg] = 1.0  # Left legs sweep to +y
            else:
                self.lateral_sign[leg] = -1.0  # Right legs sweep to -y
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (set once, remain constant)
        self.vel_world = np.array([self.base_forward_velocity, 0.0, 0.0])
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Maintain constant forward velocity throughout the motion.
        """
        # Base moves forward at constant velocity
        self.vel_world = np.array([self.base_forward_velocity, 0.0, 0.0])
        self.omega_world = np.zeros(3)
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for synchronized swizzle motion.
        
        Phase structure (two complete strokes per cycle):
        - [0.0, 0.3]: First outward sweep (propulsive)
        - [0.3, 0.5]: First inward recovery
        - [0.5, 0.8]: Second outward sweep (propulsive)
        - [0.8, 1.0]: Second inward recovery
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        lateral_sign = self.lateral_sign[leg_name]
        
        # Determine which stroke we're in and local phase within stroke
        if phase < 0.5:
            # First stroke
            stroke_phase = phase / 0.5
        else:
            # Second stroke
            stroke_phase = (phase - 0.5) / 0.5
        
        # Within each stroke: [0.0, 0.6] = outward sweep, [0.6, 1.0] = inward recovery
        if stroke_phase < 0.6:
            # Outward sweep phase (propulsive)
            local_phase = stroke_phase / 0.6
            
            # Smooth lateral motion: 0 -> max amplitude
            # Use smooth interpolation with ease-in-ease-out
            lateral_progress = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            lateral_offset = lateral_sign * self.swizzle_amplitude * lateral_progress
            
            # Backward motion in body frame during outward sweep
            # Foot moves backward relative to base to generate propulsion
            forward_offset = -self.forward_backward_amplitude * lateral_progress
            
        else:
            # Inward recovery phase
            local_phase = (stroke_phase - 0.6) / 0.4
            
            # Smooth lateral motion: max amplitude -> 0
            lateral_progress = 0.5 * (1.0 + np.cos(np.pi * local_phase))
            lateral_offset = lateral_sign * self.swizzle_amplitude * lateral_progress
            
            # Forward motion in body frame during inward recovery
            # Foot advances forward as it converges to centerline
            forward_offset = -self.forward_backward_amplitude * (1.0 - local_phase)
        
        # Apply offsets to base position
        foot = base_pos.copy()
        foot[0] += forward_offset  # Forward/backward in body frame
        foot[1] += lateral_offset  # Lateral swizzle motion
        foot[2] = base_pos[2]      # Keep z constant (ground contact)
        
        return foot