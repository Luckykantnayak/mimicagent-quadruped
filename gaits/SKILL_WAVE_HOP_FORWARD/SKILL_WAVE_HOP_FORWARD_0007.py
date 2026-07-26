from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_WAVE_HOP_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Forward hopping gait with sequential rear-to-front liftoff wave.
    
    Phase structure:
      [0.0, 0.15]: rear_liftoff - rear legs lift, front legs push
      [0.15, 0.3]: front_liftoff - front legs push off
      [0.3, 0.5]: full_flight - all legs airborne
      [0.5, 0.65]: rear_landing - rear legs contact first
      [0.65, 0.8]: front_landing - front legs contact
      [0.8, 1.0]: compression_prep - all legs compress
    
    Base motion uses velocity commands to generate forward translation
    and controlled pitch during wave propagation.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Slightly slower for dynamic hop
        
        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.step_height = 0.15  # Height during flight
        self.step_length = 0.25  # Forward displacement per hop
        self.compression_depth = 0.08  # Leg compression during prep phase
        
        # Base velocity parameters
        self.max_forward_vel = 1.2
        self.max_upward_vel = 1.5
        self.max_pitch_rate = 0.8
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocities based on phase to create wave hop dynamics.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.15: rear_liftoff
        if phase < 0.15:
            t_local = phase / 0.15
            vx = self.max_forward_vel * 0.4 * t_local
            vz = self.max_upward_vel * 0.3 * t_local
            pitch_rate = -self.max_pitch_rate * 0.5 * np.sin(np.pi * t_local)
        
        # Phase 0.15-0.3: front_liftoff
        elif phase < 0.3:
            t_local = (phase - 0.15) / 0.15
            vx = self.max_forward_vel * (0.4 + 0.6 * t_local)
            vz = self.max_upward_vel * (0.3 + 0.7 * t_local)
            pitch_rate = self.max_pitch_rate * np.sin(np.pi * t_local)
        
        # Phase 0.3-0.5: full_flight
        elif phase < 0.5:
            t_local = (phase - 0.3) / 0.2
            vx = self.max_forward_vel * (1.0 - 0.3 * t_local)
            vz = self.max_upward_vel * (1.0 - 2.0 * t_local)
            pitch_rate = 0.0
        
        # Phase 0.5-0.65: rear_landing
        elif phase < 0.65:
            t_local = (phase - 0.5) / 0.15
            vx = self.max_forward_vel * 0.7 * (1.0 - 0.5 * t_local)
            vz = -self.max_upward_vel * 0.5 * (1.0 - 0.6 * t_local)
            pitch_rate = -self.max_pitch_rate * 0.6 * t_local
        
        # Phase 0.65-0.8: front_landing
        elif phase < 0.8:
            t_local = (phase - 0.65) / 0.15
            vx = self.max_forward_vel * 0.35 * (1.0 - t_local)
            vz = -self.max_upward_vel * 0.2 * (1.0 - t_local)
            pitch_rate = self.max_pitch_rate * 0.3 * (1.0 - t_local)
        
        # Phase 0.8-1.0: compression_prep
        else:
            t_local = (phase - 0.8) / 0.2
            vx = 0.0
            vz = -0.3 * np.sin(np.pi * t_local)
            pitch_rate = 0.0
        
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
        Compute foot trajectory in body frame for each leg based on phase.
        Rear legs (RL, RR) lead the wave, front legs (FL, FR) follow.
        """
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        is_rear = leg_name.startswith('R')
        is_front = leg_name.startswith('F')
        
        # Rear legs: lift at phase 0.0, land at phase 0.5
        if is_rear:
            return self._compute_rear_leg_trajectory(foot_base, phase)
        
        # Front legs: lift at phase 0.15, land at phase 0.65
        if is_front:
            return self._compute_front_leg_trajectory(foot_base, phase)
        
        return foot_base

    def _compute_rear_leg_trajectory(self, foot_base, phase):
        """
        Rear leg trajectory: liftoff wave initiator, lands first.
        """
        foot = foot_base.copy()
        
        # Phase 0.0-0.15: rapid liftoff and retraction
        if phase < 0.15:
            t = phase / 0.15
            foot[2] += self.step_height * 0.4 * np.sin(np.pi * t)
            foot[0] += self.step_length * 0.1 * t
        
        # Phase 0.15-0.3: continue upward, tucked
        elif phase < 0.3:
            t = (phase - 0.15) / 0.15
            foot[2] += self.step_height * (0.4 + 0.4 * t)
            foot[0] += self.step_length * (0.1 + 0.2 * t)
        
        # Phase 0.3-0.5: apex and extend for landing
        elif phase < 0.5:
            t = (phase - 0.3) / 0.2
            foot[2] += self.step_height * (0.8 - 0.5 * t)
            foot[0] += self.step_length * (0.3 + 0.4 * t)
        
        # Phase 0.5-0.65: landing contact and compression
        elif phase < 0.65:
            t = (phase - 0.5) / 0.15
            foot[2] += self.step_height * 0.3 * (1.0 - t)
            foot[0] += self.step_length * 0.7
            foot[2] -= self.compression_depth * 0.3 * t
        
        # Phase 0.65-0.8: continued compression with all legs
        elif phase < 0.8:
            t = (phase - 0.65) / 0.15
            foot[0] += self.step_length * 0.7
            foot[2] -= self.compression_depth * (0.3 + 0.3 * t)
        
        # Phase 0.8-1.0: full compression prep
        else:
            t = (phase - 0.8) / 0.2
            foot[0] += self.step_length * 0.7
            foot[2] -= self.compression_depth * (0.6 + 0.4 * np.sin(np.pi * t))
        
        return foot

    def _compute_front_leg_trajectory(self, foot_base, phase):
        """
        Front leg trajectory: follows rear legs, provides main thrust.
        """
        foot = foot_base.copy()
        
        # Phase 0.0-0.15: stance, push base upward and forward
        if phase < 0.15:
            t = phase / 0.15
            foot[0] -= self.step_length * 0.3 * t
            foot[2] -= 0.02 * t  # Slight compression during push
        
        # Phase 0.15-0.3: liftoff and retraction
        elif phase < 0.3:
            t = (phase - 0.15) / 0.15
            foot[0] -= self.step_length * 0.3 * (1.0 - t)
            foot[2] += self.step_height * 0.6 * np.sin(np.pi * t)
        
        # Phase 0.3-0.5: flight, extend forward
        elif phase < 0.5:
            t = (phase - 0.3) / 0.2
            foot[2] += self.step_height * 0.6
            foot[0] += self.step_length * 0.5 * t
        
        # Phase 0.5-0.65: continue extending toward landing
        elif phase < 0.65:
            t = (phase - 0.5) / 0.15
            foot[2] += self.step_height * 0.6 * (1.0 - 0.5 * t)
            foot[0] += self.step_length * (0.5 + 0.3 * t)
        
        # Phase 0.65-0.8: landing contact and compression
        elif phase < 0.8:
            t = (phase - 0.65) / 0.15
            foot[2] += self.step_height * 0.3 * (1.0 - t)
            foot[0] += self.step_length * 0.8
            foot[2] -= self.compression_depth * 0.5 * t
        
        # Phase 0.8-1.0: full compression prep
        else:
            t = (phase - 0.8) / 0.2
            foot[0] += self.step_length * 0.8
            foot[2] -= self.compression_depth * (0.5 + 0.5 * np.sin(np.pi * t))
        
        return foot