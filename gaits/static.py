# gaits/static_gait.py
import numpy as np
from gaits.base import BaseMotionGenerator

class StaticMotionGenerator(BaseMotionGenerator):
    def __init__(self, initial_foot_positions_body, leg_names, 
                 freq=1.0, duty_ratio=0.75, step_length=0.1, step_height=0.05, style="handstand"):
        """
        Args:
            initial_foot_positions_body: dict {leg_name: [x,y,z]} in body frame
            leg_names: list of leg names (e.g., ["FL_calf", "FR_calf", "RL_calf", "RR_calf"])
            freq: gait frequency (Hz)
            duty_ratio: fraction of cycle in stance phase
            step_length: forward step length (m)
            step_height: maximum foot lift height (m)
        """
        self.leg_names = leg_names
        self.freq = freq
        self.duty = duty_ratio
        self.step_length = step_length
        self.step_height = step_height
        
        # Store foot base positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for diagonal gait pattern
        self.phase_offsets = {
            leg_names[0]: 0.0,      # FL
            leg_names[1]: np.pi,    # FR
            leg_names[2]: np.pi,    # RL
            leg_names[3]: 0.0,      # RR
        }

        # State
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, 0.0])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
        
        # Velocity commands (world frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def compute_foot_position_body_frame(self, leg_name, t):
        """
        Compute target foot position in body frame based on gait phase.
        
        Args:
            leg_name: name of the leg
            t: current time (s)
            
        Returns:
            foot_pos_body: [x, y, z] in body frame
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        return foot
