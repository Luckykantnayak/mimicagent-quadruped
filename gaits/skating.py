import numpy as np
from gaits.base import BaseMotionGenerator

class SkatingMotionGenerator(BaseMotionGenerator):
    """
    Generates walking motion by computing:
    1. Root pose from world-frame velocity commands
    2. Foot positions in body frame using gait patterns
    """
    
    def __init__(self, initial_foot_positions_body, leg_names, 
                 freq=1.0, duty_ratio=0.75, step_length=0.1, step_height=0.05,
                 push_ratio=0.2, recovery_ratio=0.2, glide_ratio=0.6, style="handstand"):
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
        
        # State
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, 0.0])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
        
        # Velocity commands (world frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

        self.push_ratio = push_ratio
        self.recovery_ratio = recovery_ratio
        self.glide_ratio = glide_ratio
        self.step_length = step_length
        self.step_height = step_height
        self._configure_style(style)

    def _configure_style(self, style):
        if style in ["handstand_sync", "front_alt"]:
            self.freeze_legs = ["RL_calf", "RR_calf"]
            self.alternating_legs = [["FL_calf"], ["FR_calf"]]
        elif style == "back_alt":
            self.freeze_legs = ["FL_calf", "FR_calf"]
            self.alternating_legs = [["RL_calf"], ["RR_calf"]]
        elif style == "diagonal_sync":
            self.freeze_legs = []
            self.alternating_legs = [["FL_calf", "RR_calf"], ["FR_calf", "RL_calf"]]
        else:
            self.freeze_legs = []
            self.alternating_legs = [["FL_calf", "RL_calf"], ["FR_calf", "RR_calf"]]

    def _compute_leg_phase(self, leg_name, t):
        if leg_name in self.freeze_legs:
            return 0.0, False
        period = 2.0 / self.freq
        time_in_period = t % period
        cycle_time = 1.0 / self.freq

        if leg_name in self.alternating_legs[0]:
            return (time_in_period / cycle_time, True) if time_in_period < cycle_time else (1.0, False)
        elif leg_name in self.alternating_legs[1]:
            return ((time_in_period - cycle_time) / cycle_time, True) if time_in_period >= cycle_time else (1.0, False)
        return 0.0, False

    def compute_foot_position_body_frame(self, leg_name, t):
        """
        Compute target foot position in body frame based on gait phase.
        
        Args:
            leg_name: name of the leg
            t: current time (s)
            
        Returns:
            foot_pos_body: [x, y, z] in body frame
        """
        phase, should_execute = self._compute_leg_phase(leg_name, t)
        foot = self.base_feet_pos_body[leg_name].copy()
        if not should_execute:
            return foot

        if phase < self.push_ratio:
            foot[0] -= self.step_length * (phase / self.push_ratio)
        elif phase < self.push_ratio + self.recovery_ratio:
            progress = (phase - self.push_ratio) / self.recovery_ratio
            angle = np.pi * progress
            foot[0] -= self.step_length * (1 - progress)
            foot[2] += self.step_height * np.sin(angle)
        return foot
