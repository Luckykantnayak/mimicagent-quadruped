from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRAB_DIAGONAL_SCUTTLE_MotionGenerator(BaseMotionGenerator):
    """
    Crab-style diagonal scuttle gait.
    
    Motion pattern:
    - Phase 0.0-0.3: First power stroke (all legs push, diagonal thrust)
    - Phase 0.3-0.5: Rapid reset (legs reposition, brief aerial phase)
    - Phase 0.5-0.8: Second amplified power stroke (increased velocity)
    - Phase 0.8-1.0: Glide and stabilize (deceleration, all feet contact)
    
    Body orientation remains perpendicular to travel direction (zero yaw rate).
    Diagonal motion achieved via equal vx and vy commands.
    Front legs sweep rearward, rear legs sweep forward during power strokes.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Leg sweep parameters
        self.front_sweep_length = 0.12  # Front legs sweep rearward
        self.rear_sweep_length = 0.12   # Rear legs sweep forward
        self.swing_height = 0.10        # Height during reset swing
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity parameters for diagonal scuttle
        self.first_stroke_vel = 0.8   # Moderate velocity for first power stroke
        self.second_stroke_vel = 1.1  # Amplified velocity (1.375x first stroke)
        self.reset_vel = 0.3          # Reduced velocity during reset
        
    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Maintains zero yaw rate while commanding diagonal velocity.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        if phase < 0.3:
            # First power stroke: moderate diagonal velocity
            progress = phase / 0.3
            vel_mag = self.first_stroke_vel * np.sin(np.pi * progress)
            vx = vel_mag
            vy = vel_mag
            
        elif phase < 0.5:
            # Rapid reset: reduced velocity, slight upward component
            progress = (phase - 0.3) / 0.2
            vel_mag = self.reset_vel * (1.0 - progress)
            vx = vel_mag
            vy = vel_mag
            vz = 0.05 * np.sin(np.pi * progress)  # Brief upward motion during aerial reset
            
        elif phase < 0.8:
            # Second amplified power stroke: higher velocity
            progress = (phase - 0.5) / 0.3
            vel_mag = self.second_stroke_vel * np.sin(np.pi * progress)
            vx = vel_mag
            vy = vel_mag
            
        else:
            # Glide and stabilize: decaying velocity
            progress = (phase - 0.8) / 0.2
            decay = 1.0 - progress
            vel_mag = self.second_stroke_vel * 0.3 * decay
            vx = vel_mag
            vy = vel_mag
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, 0.0])  # Zero yaw rate maintains body orientation
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase and leg group.
        Front legs (FL, FR) sweep rearward during power strokes.
        Rear legs (RL, RR) sweep forward during power strokes.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        
        if phase < 0.3:
            # First power stroke: stance phase with sweep
            progress = phase / 0.3
            if is_front:
                # Front legs sweep rearward (negative x)
                foot[0] -= self.front_sweep_length * progress
            else:
                # Rear legs sweep forward (positive x)
                foot[0] += self.rear_sweep_length * progress
                
        elif phase < 0.5:
            # Rapid reset: swing phase to reposition
            progress = (phase - 0.3) / 0.2
            swing_angle = np.pi * progress
            
            if is_front:
                # Front legs swing forward to reset
                x_start = self.base_feet_pos_body[leg_name][0] - self.front_sweep_length
                x_end = self.base_feet_pos_body[leg_name][0]
                foot[0] = x_start + (x_end - x_start) * progress
            else:
                # Rear legs swing rearward to reset
                x_start = self.base_feet_pos_body[leg_name][0] + self.rear_sweep_length
                x_end = self.base_feet_pos_body[leg_name][0]
                foot[0] = x_start + (x_end - x_start) * progress
            
            # Add swing height during reset
            foot[2] += self.swing_height * np.sin(swing_angle)
            
        elif phase < 0.8:
            # Second amplified power stroke: larger sweep amplitude
            progress = (phase - 0.5) / 0.3
            amplification = 1.4  # Increased amplitude for second stroke
            
            if is_front:
                # Front legs sweep rearward with amplification
                foot[0] -= self.front_sweep_length * amplification * progress
            else:
                # Rear legs sweep forward with amplification
                foot[0] += self.rear_sweep_length * amplification * progress
                
        else:
            # Glide and stabilize: return to neutral stance
            progress = (phase - 0.8) / 0.2
            
            if is_front:
                x_start = self.base_feet_pos_body[leg_name][0] - self.front_sweep_length * 1.4
                x_end = self.base_feet_pos_body[leg_name][0]
                foot[0] = x_start + (x_end - x_start) * progress
            else:
                x_start = self.base_feet_pos_body[leg_name][0] + self.rear_sweep_length * 1.4
                x_end = self.base_feet_pos_body[leg_name][0]
                foot[0] = x_start + (x_end - x_start) * progress
        
        return foot