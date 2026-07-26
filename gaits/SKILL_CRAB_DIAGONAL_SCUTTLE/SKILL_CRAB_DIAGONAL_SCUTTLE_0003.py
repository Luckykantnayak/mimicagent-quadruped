from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRAB_DIAGONAL_SCUTTLE_MotionGenerator(BaseMotionGenerator):
    """
    Crab-like diagonal scuttle motion with body oriented perpendicular to travel direction.
    
    Motion structure:
    - Phase 0.0-0.3: First scuttle stroke (all legs in contact, coordinated sweep)
    - Phase 0.3-0.5: Rapid leg reset (minimal contact, momentum carry)
    - Phase 0.5-0.8: Second powerful scuttle stroke (amplified sweep)
    - Phase 0.8-1.0: Glide and stabilization (deceleration to rest/cruise)
    
    Coordination:
    - Front legs (FL, FR) sweep rearward during strokes
    - Rear legs (RL, RR) sweep forward during strokes (anti-phase with front)
    - Body yaw locked at ~90° to achieve sideways orientation relative to travel
    - Diagonal velocity vector (forward-right) achieved through leg coordination
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Stroke parameters - reduced to prevent joint limit violations
        self.first_stroke_amplitude = 0.10
        self.second_stroke_amplitude = 0.12
        self.sweep_lateral_offset = 0.025
        self.swing_height = 0.05
        
        # Rear leg amplitude scaling factor (rear legs have more restricted forward reach)
        self.rear_leg_amplitude_factor = 0.75
        
        # Velocity parameters for diagonal motion
        self.vx_first_stroke = 0.5
        self.vy_first_stroke = 0.5
        self.vx_second_stroke = 0.85
        self.vy_second_stroke = 0.85
        self.reset_vz = 0.04
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        
        # Initialize with 90° yaw for sideways crab orientation
        self.root_quat = euler_to_quat(0.0, 0.0, np.pi / 2)
        
        # Command state
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent diagonal velocity profile.
        Body yaw remains locked at 90° throughout.
        """
        
        if phase < 0.3:
            # First scuttle stroke: build diagonal velocity
            progress = phase / 0.3
            smooth_ramp = 0.5 * (1 - np.cos(np.pi * progress))
            vx = self.vx_first_stroke * smooth_ramp
            vy = self.vy_first_stroke * smooth_ramp
            vz = 0.0
            
        elif phase < 0.5:
            # Rapid reset: coast on momentum with slight vertical hop
            progress = (phase - 0.3) / 0.2
            decay = 1.0 - 0.4 * progress
            vx = self.vx_first_stroke * decay
            vy = self.vy_first_stroke * decay
            vz = self.reset_vz * np.sin(np.pi * progress)
            
        elif phase < 0.8:
            # Second powerful stroke: peak diagonal velocity
            progress = (phase - 0.5) / 0.3
            smooth_profile = np.sin(np.pi * progress)
            vx = self.vx_second_stroke * smooth_profile
            vy = self.vy_second_stroke * smooth_profile
            vz = 0.0
            
        else:
            # Glide and stabilize: smooth deceleration
            progress = (phase - 0.8) / 0.2
            decay = np.cos(0.5 * np.pi * progress)
            vx = self.vx_second_stroke * 0.6 * decay
            vy = self.vy_second_stroke * 0.6 * decay
            vz = -0.015 * progress
        
        # Yaw rate locked at zero to maintain perpendicular body orientation
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot trajectory in body frame.
        
        Front legs (FL, FR): sweep rearward during strokes, forward during reset
        Rear legs (RL, RR): sweep forward during strokes, rearward during reset
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Apply rear leg amplitude reduction to respect workspace constraints
        leg_amplitude_factor = 1.0 if is_front else self.rear_leg_amplitude_factor
        
        if phase < 0.3:
            # First scuttle stroke (stance phase)
            progress = phase / 0.3
            smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))
            
            if is_front:
                # Front legs sweep rearward
                dx = -self.first_stroke_amplitude * smooth_progress
            else:
                # Rear legs sweep forward with reduced amplitude
                dx = self.first_stroke_amplitude * leg_amplitude_factor * smooth_progress
            
            # Slight lateral component for diagonal thrust
            dy = self.sweep_lateral_offset * smooth_progress * (1 if is_left else -1)
            
            foot[0] += dx
            foot[1] += dy
            
        elif phase < 0.5:
            # Rapid reset (swing phase)
            progress = (phase - 0.3) / 0.2
            smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))
            
            if is_front:
                # Front legs swing forward to reset
                dx_start = -self.first_stroke_amplitude
                dx_end = 0.0
                dx = dx_start + (dx_end - dx_start) * smooth_progress
            else:
                # Rear legs swing rearward to reset
                dx_start = self.first_stroke_amplitude * leg_amplitude_factor
                dx_end = 0.0
                dx = dx_start + (dx_end - dx_start) * smooth_progress
            
            dy_start = self.sweep_lateral_offset * (1 if is_left else -1)
            dy = dy_start * (1 - smooth_progress)
            
            # High-speed swing with ground clearance
            dz = self.swing_height * np.sin(np.pi * progress)
            
            foot[0] += dx
            foot[1] += dy
            foot[2] += dz
            
        elif phase < 0.8:
            # Second powerful stroke (stance phase with ramped amplitude)
            progress = (phase - 0.5) / 0.3
            
            # Smooth amplitude ramp to reduce peak joint demands
            amplitude_envelope = np.sin(np.pi * progress)
            smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))
            
            if is_front:
                # Front legs sweep rearward with increased power
                dx = -self.second_stroke_amplitude * amplitude_envelope * smooth_progress
            else:
                # Rear legs sweep forward with increased power and reduced amplitude
                dx = self.second_stroke_amplitude * leg_amplitude_factor * amplitude_envelope * smooth_progress
            
            # Reduced lateral offset scaling to minimize diagonal excursion
            lateral_scale = 1.1
            dy = self.sweep_lateral_offset * lateral_scale * smooth_progress * (1 if is_left else -1)
            
            foot[0] += dx
            foot[1] += dy
            
        else:
            # Glide and stabilize (return to nominal stance)
            progress = (phase - 0.8) / 0.2
            smooth_progress = 0.5 * (1 + np.cos(np.pi * (1 - progress)))
            
            # Calculate peak amplitude at end of second stroke for smooth transition
            peak_amplitude_envelope = np.sin(np.pi * 1.0)
            
            if is_front:
                dx_start = -self.second_stroke_amplitude * peak_amplitude_envelope
                dx_end = 0.0
            else:
                dx_start = self.second_stroke_amplitude * leg_amplitude_factor * peak_amplitude_envelope
                dx_end = 0.0
            
            dx = dx_start + (dx_end - dx_start) * smooth_progress
            
            lateral_scale = 1.1
            dy_start = self.sweep_lateral_offset * lateral_scale * (1 if is_left else -1)
            dy = dy_start * (1 - smooth_progress)
            
            foot[0] += dx
            foot[1] += dy
        
        return foot