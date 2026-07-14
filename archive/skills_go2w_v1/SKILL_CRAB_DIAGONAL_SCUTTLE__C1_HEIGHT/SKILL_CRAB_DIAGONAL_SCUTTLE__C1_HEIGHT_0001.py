from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRAB_DIAGONAL_SCUTTLE_MotionGenerator(BaseMotionGenerator):
    """
    Crab-style diagonal scuttle gait.
    
    Robot maintains sideways body orientation while scuttling diagonally forward-right.
    Front legs sweep rearward, rear legs sweep forward in coordinated strokes.
    
    Phase structure:
    - [0.0, 0.3]: First power stroke (all legs in contact, diagonal thrust)
    - [0.3, 0.5]: Rapid reset (legs reposition, brief aerial phase)
    - [0.5, 0.8]: Second amplified power stroke (increased velocity)
    - [0.8, 1.0]: Glide and stabilize (coast on momentum)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Leg sweep parameters
        self.front_sweep_length = 0.12  # Front legs sweep rearward
        self.rear_sweep_length = 0.12   # Rear legs sweep forward
        self.swing_height = 0.10        # Height during reset swing
        
        # Velocity parameters for diagonal motion
        self.vx_stroke1 = 0.6   # Forward velocity during first stroke
        self.vy_stroke1 = 0.6   # Rightward velocity during first stroke (equal for 45° diagonal)
        self.vx_stroke2 = 0.9   # Amplified forward velocity for second stroke (1.5x)
        self.vy_stroke2 = 0.9   # Amplified rightward velocity for second stroke
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (world frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Maintains zero yaw rate (sideways orientation) while commanding diagonal velocity.
        """
        
        if 0.0 <= phase < 0.3:
            # First power stroke: moderate diagonal velocity
            vx = self.vx_stroke1
            vy = self.vy_stroke1
            
        elif 0.3 <= phase < 0.5:
            # Rapid reset: reduced velocity to allow leg repositioning
            # Slight backward/leftward bias to counteract drift
            progress = (phase - 0.3) / 0.2
            vx = self.vx_stroke1 * (1.0 - progress) - 0.2 * progress
            vy = self.vy_stroke1 * (1.0 - progress) - 0.2 * progress
            
        elif 0.5 <= phase < 0.8:
            # Second amplified power stroke: increased diagonal velocity
            vx = self.vx_stroke2
            vy = self.vy_stroke2
            
        else:  # 0.8 <= phase < 1.0
            # Glide and stabilize: decay velocity to zero
            progress = (phase - 0.8) / 0.2
            vx = self.vx_stroke2 * (1.0 - progress)
            vy = self.vy_stroke2 * (1.0 - progress)
        
        # Set velocity commands (world frame)
        self.vel_world = np.array([vx, vy, 0.0])
        
        # Maintain zero yaw rate to keep sideways orientation
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
        Compute foot position in body frame based on phase and leg group.
        
        Front legs (FL, FR): sweep rearward during power strokes, forward during reset
        Rear legs (RL, RR): sweep forward during power strokes, rearward during reset
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if front or rear leg
        is_front = leg_name.startswith('F')
        
        if 0.0 <= phase < 0.3:
            # First power stroke: all legs in contact
            progress = phase / 0.3
            
            if is_front:
                # Front legs sweep rearward (negative x)
                foot[0] -= self.front_sweep_length * progress
            else:
                # Rear legs sweep forward (positive x)
                foot[0] += self.rear_sweep_length * progress
                
        elif 0.3 <= phase < 0.5:
            # Rapid reset: swing legs back to starting position
            progress = (phase - 0.3) / 0.2
            
            if is_front:
                # Front legs swing forward with arc
                x_start = -self.front_sweep_length
                x_end = 0.0
                foot[0] += x_start + (x_end - x_start) * progress
                # Arc trajectory for swing
                foot[2] += self.swing_height * np.sin(np.pi * progress)
            else:
                # Rear legs swing rearward with arc
                x_start = self.rear_sweep_length
                x_end = 0.0
                foot[0] += x_start + (x_end - x_start) * progress
                foot[2] += self.swing_height * np.sin(np.pi * progress)
                
        elif 0.5 <= phase < 0.8:
            # Second amplified power stroke: larger displacement
            progress = (phase - 0.5) / 0.3
            amplification = 1.3  # Slightly larger sweep for second stroke
            
            if is_front:
                # Front legs sweep rearward with amplification
                foot[0] -= self.front_sweep_length * amplification * progress
            else:
                # Rear legs sweep forward with amplification
                foot[0] += self.rear_sweep_length * amplification * progress
                
        else:  # 0.8 <= phase < 1.0
            # Glide and stabilize: return to neutral stance
            progress = (phase - 0.8) / 0.2
            amplification = 1.3
            
            if is_front:
                # Smoothly return from amplified rearward position to neutral
                x_start = -self.front_sweep_length * amplification
                x_end = 0.0
                foot[0] += x_start + (x_end - x_start) * progress
            else:
                # Smoothly return from amplified forward position to neutral
                x_start = self.rear_sweep_length * amplification
                x_end = 0.0
                foot[0] += x_start + (x_end - x_start) * progress
        
        return foot