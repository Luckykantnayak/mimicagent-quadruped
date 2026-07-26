from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_BOUND_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Bound gait with alternating front and rear leg pairs.
    
    Phase structure:
    - [0.0, 0.45]: Front stance, rear swing
    - [0.45, 0.55]: Front-to-rear transition (all feet grounded)
    - [0.55, 0.95]: Rear stance, front swing
    - [0.95, 1.0]: Rear-to-front transition (all feet grounded)
    
    Front legs (FL, FR) move in synchrony; rear legs (RL, RR) move in synchrony.
    Front and rear pairs are 180° out of phase.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8
        
        # Gait parameters
        self.step_length = 0.15
        self.step_height = 0.10
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base velocity parameters
        self.forward_velocity = 0.6
        self.vz_amplitude = 0.05
        self.pitch_rate_amplitude = 0.3
        
        # Phase assignments for front vs rear pairs
        # Front legs: swing during [0.55, 0.95], stance during [0.0, 0.45]
        # Rear legs: swing during [0.0, 0.45], stance during [0.55, 0.95]
        self.front_legs = [leg for leg in leg_names if leg.startswith('F')]
        self.rear_legs = [leg for leg in leg_names if leg.startswith('R')]

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on current phase.
        
        Front stance [0.0, 0.45]: pitch down, vz slightly up
        Transition [0.45, 0.55]: pitch reversal, vz transitions
        Rear stance [0.55, 0.95]: pitch up, vz slightly down
        Transition [0.95, 1.0]: pitch reversal, vz transitions
        """
        
        # Forward velocity is constant
        vx = self.forward_velocity
        vy = 0.0
        
        # Vertical velocity and pitch rate vary with phase
        if phase < 0.45:
            # Front stance phase
            phase_local = phase / 0.45
            vz = self.vz_amplitude * np.sin(np.pi * phase_local)
            pitch_rate = -self.pitch_rate_amplitude * np.sin(np.pi * phase_local)
            
        elif phase < 0.55:
            # Front-to-rear transition
            phase_local = (phase - 0.45) / 0.1
            vz = self.vz_amplitude * (1.0 - 2.0 * phase_local)
            pitch_rate = self.pitch_rate_amplitude * (2.0 * phase_local - 1.0) * 2.0
            
        elif phase < 0.95:
            # Rear stance phase
            phase_local = (phase - 0.55) / 0.4
            vz = -self.vz_amplitude * np.sin(np.pi * phase_local)
            pitch_rate = self.pitch_rate_amplitude * np.sin(np.pi * phase_local)
            
        else:
            # Rear-to-front transition
            phase_local = (phase - 0.95) / 0.05
            vz = -self.vz_amplitude * (1.0 - 2.0 * phase_local)
            pitch_rate = -self.pitch_rate_amplitude * (2.0 * phase_local - 1.0) * 2.0
        
        # Set velocities in world frame
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
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
        
        Front legs (FL, FR):
        - Stance: [0.0, 0.55] (includes transition)
        - Swing: [0.55, 0.95]
        - Early stance: [0.95, 1.0]
        
        Rear legs (RL, RR):
        - Swing: [0.0, 0.45]
        - Early stance: [0.45, 0.55]
        - Stance: [0.55, 1.0] (includes transition)
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_name in self.front_legs:
            # Front leg motion
            if phase < 0.55:
                # Stance phase: foot sweeps backward in body frame
                progress = phase / 0.55
                foot[0] += self.step_length * (0.5 - progress)
                
            elif phase < 0.95:
                # Swing phase: foot arcs forward
                progress = (phase - 0.55) / 0.4
                angle = np.pi * progress
                foot[0] += self.step_length * (progress - 0.5)
                foot[2] += self.step_height * np.sin(angle)
                
            else:
                # Transition: early stance, foot planted forward
                foot[0] += self.step_length * 0.5
                
        else:
            # Rear leg motion (phase-opposite to front)
            if phase < 0.45:
                # Swing phase: foot arcs forward
                progress = phase / 0.45
                angle = np.pi * progress
                foot[0] += self.step_length * (progress - 0.5)
                foot[2] += self.step_height * np.sin(angle)
                
            elif phase < 0.55:
                # Transition: early stance, foot planted forward
                foot[0] += self.step_length * 0.5
                
            else:
                # Stance phase: foot sweeps backward in body frame
                progress = (phase - 0.55) / 0.45
                foot[0] += self.step_length * (0.5 - progress)
        
        return foot