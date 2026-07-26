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
        self.first_stroke_speed = 0.8   # Moderate speed for first power stroke
        self.second_stroke_speed = 1.1  # Amplified speed (1.375x first stroke)
        self.reset_speed = 0.3          # Reduced speed during reset
        
    def update_base_motion(self, phase, dt):
        """
        Update base velocity to achieve diagonal scuttle motion.
        Body orientation remains fixed (zero yaw rate).
        Diagonal motion via equal vx and vy components.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        # Phase 0.0-0.3: First power stroke
        if phase < 0.3:
            progress = phase / 0.3
            # Smooth acceleration using sine
            speed_factor = np.sin(progress * np.pi / 2)
            vx = self.first_stroke_speed * speed_factor
            vy = self.first_stroke_speed * speed_factor
            
        # Phase 0.3-0.5: Rapid reset
        elif phase < 0.5:
            progress = (phase - 0.3) / 0.2
            # Decelerate during reset
            speed_factor = np.cos(progress * np.pi / 2)
            vx = self.reset_speed * speed_factor
            vy = self.reset_speed * speed_factor
            # Brief upward velocity during aerial reset phase
            vz = 0.05 * np.sin(progress * np.pi)
            
        # Phase 0.5-0.8: Second amplified power stroke
        elif phase < 0.8:
            progress = (phase - 0.5) / 0.3
            # Stronger acceleration for amplified stroke
            speed_factor = np.sin(progress * np.pi / 2)
            vx = self.second_stroke_speed * speed_factor
            vy = self.second_stroke_speed * speed_factor
            
        # Phase 0.8-1.0: Glide and stabilize
        else:
            progress = (phase - 0.8) / 0.2
            # Exponential decay for smooth glide
            decay_factor = np.exp(-5.0 * progress)
            vx = self.second_stroke_speed * 0.5 * decay_factor
            vy = self.second_stroke_speed * 0.5 * decay_factor
        
        # Set velocities (zero angular rates to maintain body orientation)
        self.vel_world = np.array([vx, vy, vz])
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
        
        Front legs (FL, FR): Sweep rearward during power, forward during reset
        Rear legs (RL, RR): Sweep forward during power, rearward during reset
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        
        # Phase 0.0-0.3: First power stroke
        if phase < 0.3:
            progress = phase / 0.3
            if is_front:
                # Front legs sweep rearward (negative x)
                foot[0] -= self.front_sweep_length * progress
            else:
                # Rear legs sweep forward (positive x)
                foot[0] += self.rear_sweep_length * progress
                
        # Phase 0.3-0.5: Rapid reset (swing phase)
        elif phase < 0.5:
            progress = (phase - 0.3) / 0.2
            if is_front:
                # Front legs swing forward to reset
                # Start from rearmost position, swing forward with arc
                x_start = -self.front_sweep_length
                x_end = self.front_sweep_length * 0.3  # Overshoot slightly for next stroke
                foot[0] += x_start + (x_end - x_start) * progress
                # Arc trajectory during swing
                foot[2] += self.swing_height * np.sin(progress * np.pi)
            else:
                # Rear legs swing rearward to reset
                x_start = self.rear_sweep_length
                x_end = -self.rear_sweep_length * 0.3
                foot[0] += x_start + (x_end - x_start) * progress
                foot[2] += self.swing_height * np.sin(progress * np.pi)
                
        # Phase 0.5-0.8: Second amplified power stroke
        elif phase < 0.8:
            progress = (phase - 0.5) / 0.3
            # Amplified stroke with 1.3x displacement
            amplification = 1.3
            if is_front:
                # Start from forward position, sweep rearward with greater amplitude
                x_start = self.front_sweep_length * 0.3
                foot[0] += x_start - self.front_sweep_length * amplification * progress
            else:
                # Start from rear position, sweep forward with greater amplitude
                x_start = -self.rear_sweep_length * 0.3
                foot[0] += x_start + self.rear_sweep_length * amplification * progress
                
        # Phase 0.8-1.0: Glide and stabilize
        else:
            progress = (phase - 0.8) / 0.2
            # Smoothly return to neutral stance
            if is_front:
                # Interpolate from end of second stroke back to neutral
                x_current = self.front_sweep_length * 0.3 - self.front_sweep_length * 1.3
                foot[0] += x_current * (1.0 - progress)
            else:
                x_current = -self.rear_sweep_length * 0.3 + self.rear_sweep_length * 1.3
                foot[0] += x_current * (1.0 - progress)
        
        return foot