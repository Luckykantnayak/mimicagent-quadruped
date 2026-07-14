from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRAB_DIAGONAL_SCUTTLE_MotionGenerator(BaseMotionGenerator):
    """
    Crab diagonal scuttle gait with body oriented sideways (~90° yaw).
    
    Motion phases:
    - Phase 0.0–0.3: First scuttle stroke (moderate amplitude)
    - Phase 0.3–0.5: Rapid reset
    - Phase 0.5–0.8: Second scuttle stroke (increased amplitude)
    - Phase 0.8–1.0: Glide and stabilize
    
    All legs remain in ground contact throughout cycle.
    Front legs sweep rearward, rear legs sweep forward in body frame.
    Base moves diagonally forward-right in world frame via combined vx and vy in body frame.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Leg sweep parameters
        self.first_stroke_amplitude = 0.12  # Moderate sweep for first stroke
        self.second_stroke_amplitude = 0.17  # ~1.4x larger for second stroke
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        # Initialize body orientation to 90° yaw (sideways)
        self.root_quat = euler_to_quat(0.0, 0.0, np.pi / 2)
        
        # Base velocity parameters (in body frame)
        # Diagonal motion: both x (forward) and y (rightward) components
        self.vx_first_stroke = 0.4
        self.vy_first_stroke = 0.5
        self.vx_second_stroke = 0.6
        self.vy_second_stroke = 0.7
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Body moves diagonally forward-right via combined vx and vy in body frame.
        Yaw rate remains zero to maintain sideways orientation.
        """
        vx_body = 0.0
        vy_body = 0.0
        
        if 0.0 <= phase < 0.3:
            # First scuttle stroke: moderate diagonal velocity
            progress = phase / 0.3
            vx_body = self.vx_first_stroke * np.sin(np.pi * progress)
            vy_body = self.vy_first_stroke * np.sin(np.pi * progress)
            
        elif 0.3 <= phase < 0.5:
            # Rapid reset: decelerate to near zero
            progress = (phase - 0.3) / 0.2
            decay = np.cos(np.pi * progress * 0.5)
            vx_body = self.vx_first_stroke * 0.2 * decay
            vy_body = self.vy_first_stroke * 0.2 * decay
            
        elif 0.5 <= phase < 0.8:
            # Second scuttle stroke: increased diagonal velocity
            progress = (phase - 0.5) / 0.3
            vx_body = self.vx_second_stroke * np.sin(np.pi * progress)
            vy_body = self.vy_second_stroke * np.sin(np.pi * progress)
            
        elif 0.8 <= phase <= 1.0:
            # Glide and stabilize: coast on momentum
            progress = (phase - 0.8) / 0.2
            decay = np.exp(-5.0 * progress)
            vx_body = self.vx_second_stroke * 0.3 * decay
            vy_body = self.vy_second_stroke * 0.3 * decay
        
        # Convert body frame velocities to world frame
        R = quat_to_rotation_matrix(self.root_quat)
        vel_body = np.array([vx_body, vy_body, 0.0])
        self.vel_world = R @ vel_body
        
        # Zero angular velocity to maintain sideways orientation
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
        
        Front legs (FL, FR): sweep rearward (positive x to negative x)
        Rear legs (RL, RR): sweep forward (negative x to positive x)
        
        All legs remain on ground (minimal z motion).
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if front or rear leg
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        sweep_offset_x = 0.0
        
        if 0.0 <= phase < 0.3:
            # First scuttle stroke
            progress = phase / 0.3
            amplitude = self.first_stroke_amplitude
            
            if is_front:
                # Front legs sweep rearward: start at +amplitude, end at -amplitude
                sweep_offset_x = amplitude * (1.0 - 2.0 * progress)
            else:
                # Rear legs sweep forward: start at -amplitude, end at +amplitude
                sweep_offset_x = amplitude * (-1.0 + 2.0 * progress)
                
        elif 0.3 <= phase < 0.5:
            # Rapid reset: return to starting position
            progress = (phase - 0.3) / 0.2
            amplitude = self.first_stroke_amplitude
            
            if is_front:
                # Front legs: move from -amplitude back to +amplitude
                sweep_offset_x = amplitude * (-1.0 + 2.0 * progress)
            else:
                # Rear legs: move from +amplitude back to -amplitude
                sweep_offset_x = amplitude * (1.0 - 2.0 * progress)
                
        elif 0.5 <= phase < 0.8:
            # Second scuttle stroke with increased amplitude
            progress = (phase - 0.5) / 0.3
            amplitude = self.second_stroke_amplitude
            
            if is_front:
                # Front legs sweep rearward
                sweep_offset_x = amplitude * (1.0 - 2.0 * progress)
            else:
                # Rear legs sweep forward
                sweep_offset_x = amplitude * (-1.0 + 2.0 * progress)
                
        elif 0.8 <= phase <= 1.0:
            # Glide and stabilize: hold steady stance
            amplitude = self.second_stroke_amplitude
            
            if is_front:
                # Front legs hold at rear position
                sweep_offset_x = -amplitude
            else:
                # Rear legs hold at forward position
                sweep_offset_x = amplitude
        
        # Apply sweep offset in x direction (body frame)
        foot_pos = base_pos.copy()
        foot_pos[0] += sweep_offset_x
        
        # Maintain ground contact (z remains at base level)
        # No vertical motion needed for scuttling
        
        return foot_pos