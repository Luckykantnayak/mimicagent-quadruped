from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRAB_DIAGONAL_SCUTTLE_MotionGenerator(BaseMotionGenerator):
    """
    Crab-like diagonal scuttling gait with sideways body orientation.
    
    Motion cycle:
    - Phase 0.0-0.3: First power stroke (all legs push, diagonal thrust)
    - Phase 0.3-0.5: Rapid reset (legs reposition quickly)
    - Phase 0.5-0.8: Second power stroke (amplified diagonal thrust)
    - Phase 0.8-1.0: Glide and stabilization (deceleration)
    
    Body maintains constant sideways orientation while traveling diagonally forward-right.
    Front legs sweep rearward, rear legs sweep forward during power strokes.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Leg motion parameters
        self.sweep_length_first = 0.12  # First stroke sweep amplitude
        self.sweep_length_second = 0.18  # Second stroke sweep amplitude (amplified)
        self.reset_height = 0.10  # Height during rapid reset swing
        self.stance_height_offset = 0.0  # Ground contact offset
        
        # Base velocity parameters (body frame)
        # First power stroke velocities
        self.vx_first = 0.4  # Forward velocity during first stroke
        self.vy_first = 0.4  # Rightward velocity during first stroke
        
        # Second power stroke velocities (amplified)
        self.vx_second = 0.7  # Forward velocity during second stroke
        self.vy_second = 0.7  # Rightward velocity during second stroke
        
        # Reset phase velocities (maintain momentum)
        self.vx_reset = 0.3
        self.vy_reset = 0.3
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Body orientation remains constant (zero yaw rate).
        Diagonal motion achieved via simultaneous vx and vy commands.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        if 0.0 <= phase < 0.3:
            # First power stroke: diagonal acceleration
            progress = phase / 0.3
            vx = self.vx_first * np.sin(np.pi * progress)
            vy = self.vy_first * np.sin(np.pi * progress)
            vz = -0.02  # Slight downward for ground contact
            
        elif 0.3 <= phase < 0.5:
            # Rapid reset: maintain momentum
            vx = self.vx_reset
            vy = self.vy_reset
            vz = 0.01  # Slight upward to reduce drag
            
        elif 0.5 <= phase < 0.8:
            # Second power stroke: amplified diagonal acceleration
            progress = (phase - 0.5) / 0.3
            vx = self.vx_second * np.sin(np.pi * progress)
            vy = self.vy_second * np.sin(np.pi * progress)
            vz = -0.02  # Slight downward for ground contact
            
        else:  # 0.8 <= phase < 1.0
            # Glide and stabilization: smooth deceleration
            progress = (phase - 0.8) / 0.2
            decay = np.cos(np.pi * progress / 2)  # Smooth decay from 1 to 0
            vx = self.vx_reset * decay * 0.5
            vy = self.vy_reset * decay * 0.5
            vz = 0.0
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, vy, vz])
        
        # Zero angular velocity to maintain constant body orientation
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
        
        Front legs (FL, FR): sweep rearward during power strokes
        Rear legs (RL, RR): sweep forward during power strokes
        All legs: rapid reset during phase 0.3-0.5
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if front or rear leg
        is_front = leg_name.startswith('F')
        
        if 0.0 <= phase < 0.3:
            # First power stroke: stance phase with sweep
            progress = phase / 0.3
            if is_front:
                # Front legs sweep rearward (negative x)
                foot[0] += self.sweep_length_first * (0.5 - progress)
            else:
                # Rear legs sweep forward (positive x)
                foot[0] += self.sweep_length_first * (progress - 0.5)
            foot[2] += self.stance_height_offset
            
        elif 0.3 <= phase < 0.5:
            # Rapid reset: swing phase with high clearance
            progress = (phase - 0.3) / 0.2
            if is_front:
                # Front legs return forward
                foot[0] += self.sweep_length_first * (-0.5 + progress)
            else:
                # Rear legs return rearward
                foot[0] += self.sweep_length_first * (0.5 - progress)
            # High arc for rapid repositioning
            foot[2] += self.reset_height * np.sin(np.pi * progress)
            
        elif 0.5 <= phase < 0.8:
            # Second power stroke: amplified stance phase
            progress = (phase - 0.5) / 0.3
            if is_front:
                # Front legs sweep rearward with larger amplitude
                foot[0] += self.sweep_length_second * (0.5 - progress)
            else:
                # Rear legs sweep forward with larger amplitude
                foot[0] += self.sweep_length_second * (progress - 0.5)
            foot[2] += self.stance_height_offset
            
        else:  # 0.8 <= phase < 1.0
            # Glide and stabilization: settle into neutral stance
            progress = (phase - 0.8) / 0.2
            if is_front:
                # Smoothly return to neutral from end of second stroke
                start_offset = -self.sweep_length_second * 0.5
                foot[0] += start_offset * (1.0 - progress)
            else:
                start_offset = self.sweep_length_second * 0.5
                foot[0] += start_offset * (1.0 - progress)
            foot[2] += self.stance_height_offset
        
        return foot