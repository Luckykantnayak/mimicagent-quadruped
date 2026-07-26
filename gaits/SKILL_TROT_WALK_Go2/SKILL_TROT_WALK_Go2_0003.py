from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_TROT_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal trot gait for stable forward locomotion.
    
    - Diagonal pairs (FL+RR) and (FR+RL) alternate between stance and swing
    - FL+RR stance in phase [0.0, 0.5], swing in [0.5, 1.0]
    - FR+RL swing in phase [0.0, 0.5], stance in [0.5, 1.0]
    - Base moves forward with constant velocity, zero lateral/vertical velocity
    - Swing trajectories are smooth parabolic arcs in body frame
    - Stance feet move rearward in body frame as base advances
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Gait timing: 50% duty cycle per leg (diagonal trot)
        self.duty_cycle = 0.5
        
        # Step geometry
        self.step_length = 0.20  # Forward reach in body frame (m)
        self.step_height = 0.06   # Peak swing clearance (m)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for diagonal pairing
        # FL and RR: stance [0.0, 0.5], swing [0.5, 1.0] => offset 0.0
        # FR and RL: swing [0.0, 0.5], stance [0.5, 1.0] => offset 0.5
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.phase_offsets[leg] = 0.5
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Forward velocity (m/s)
        self.forward_velocity = 0.4

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity.
        No lateral, vertical, or rotational motion.
        """
        vx = self.forward_velocity
        vy = 0.0
        vz = 0.0
        
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame.
        
        Stance phase (leg-local phase [0.0, 0.5]):
          - Foot moves linearly rearward in body frame from +step_length/2 to -step_length/2
          - Z remains at base height (contact with ground)
        
        Swing phase (leg-local phase [0.5, 1.0]):
          - Foot lifts, arcs forward, and descends
          - X moves forward from -step_length/2 to +step_length/2
          - Z follows parabolic trajectory with peak at mid-swing
        """
        # Compute leg-local phase using phase offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Get base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_phase < self.duty_cycle:
            # Stance phase: foot moves rearward in body frame
            # progress goes from 0 to 1 over stance
            progress = leg_phase / self.duty_cycle
            # Start at +step_length/2, end at -step_length/2
            foot[0] += self.step_length * (0.5 - progress)
            # Z remains at base height (no change)
        else:
            # Swing phase: foot arcs forward and upward
            # progress goes from 0 to 1 over swing
            progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)
            # Start at -step_length/2, end at +step_length/2
            foot[0] += self.step_length * (progress - 0.5)
            # Parabolic arc: peak at progress = 0.5
            swing_angle = np.pi * progress
            foot[2] += self.step_height * np.sin(swing_angle)
        
        return foot