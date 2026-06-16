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
        self.freq = 0.8
        
        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - reduced for joint limit safety
        self.step_height = 0.10  # Reduced from 0.15 to limit apex height
        self.step_length = 0.16  # Reduced from 0.25 to prevent overextension
        self.compression_depth = 0.05  # Reduced from 0.08 for joint safety
        
        # Base velocity parameters - reduced to control apex height
        self.max_forward_vel = 0.9  # Reduced from 1.2
        self.max_upward_vel = 1.0  # Reduced from 1.5 to stay within envelope
        self.max_pitch_rate = 0.6  # Reduced from 0.8
        
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
        Reduced velocity magnitudes to prevent base height violations.
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
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t_local))  # Smooth ramp
            vx = self.max_forward_vel * 0.3 * smooth_t
            vz = self.max_upward_vel * 0.25 * smooth_t
            pitch_rate = -self.max_pitch_rate * 0.4 * np.sin(np.pi * t_local)
        
        # Phase 0.15-0.3: front_liftoff
        elif phase < 0.3:
            t_local = (phase - 0.15) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t_local))
            vx = self.max_forward_vel * (0.3 + 0.5 * smooth_t)
            vz = self.max_upward_vel * (0.25 + 0.55 * smooth_t)
            pitch_rate = self.max_pitch_rate * 0.5 * np.sin(np.pi * t_local)
        
        # Phase 0.3-0.5: full_flight (apex, decelerate upward velocity)
        elif phase < 0.5:
            t_local = (phase - 0.3) / 0.2
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t_local))
            vx = self.max_forward_vel * 0.8 * (1.0 - 0.2 * smooth_t)
            vz = self.max_upward_vel * 0.8 * (1.0 - 2.0 * smooth_t)  # Transition through zero
            pitch_rate = 0.0
        
        # Phase 0.5-0.65: rear_landing
        elif phase < 0.65:
            t_local = (phase - 0.5) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t_local))
            vx = self.max_forward_vel * 0.6 * (1.0 - 0.4 * smooth_t)
            vz = -self.max_upward_vel * 0.4 * (1.0 - 0.5 * smooth_t)
            pitch_rate = -self.max_pitch_rate * 0.5 * smooth_t
        
        # Phase 0.65-0.8: front_landing
        elif phase < 0.8:
            t_local = (phase - 0.65) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t_local))
            vx = self.max_forward_vel * 0.4 * (1.0 - smooth_t)
            vz = -self.max_upward_vel * 0.2 * (1.0 - smooth_t)
            pitch_rate = self.max_pitch_rate * 0.3 * np.sin(np.pi * smooth_t)
        
        # Phase 0.8-1.0: compression_prep
        else:
            t_local = (phase - 0.8) / 0.2
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t_local))
            vx = 0.0
            vz = -0.2 * np.sin(np.pi * smooth_t)
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
        Reduced forward extension during landing to prevent joint limits and ground penetration.
        """
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        is_rear = leg_name.startswith('R')
        is_front = leg_name.startswith('F')
        
        if is_rear:
            return self._compute_rear_leg_trajectory(foot_base, phase)
        
        if is_front:
            return self._compute_front_leg_trajectory(foot_base, phase)
        
        return foot_base

    def _compute_rear_leg_trajectory(self, foot_base, phase):
        """
        Rear leg trajectory: liftoff wave initiator, lands first.
        Reduced forward extension to prevent joint violations.
        """
        foot = foot_base.copy()
        
        # Phase 0.0-0.15: rapid liftoff and retraction
        if phase < 0.15:
            t = phase / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[2] += self.step_height * 0.3 * smooth_t
            foot[0] += self.step_length * 0.1 * smooth_t
        
        # Phase 0.15-0.3: continue upward, tucked
        elif phase < 0.3:
            t = (phase - 0.15) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[2] += self.step_height * (0.3 + 0.5 * smooth_t)
            foot[0] += self.step_length * (0.1 + 0.15 * smooth_t)
        
        # Phase 0.3-0.5: apex and extend for landing
        elif phase < 0.5:
            t = (phase - 0.3) / 0.2
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[2] += self.step_height * (0.8 - 0.3 * smooth_t)
            foot[0] += self.step_length * (0.25 + 0.15 * smooth_t)
        
        # Phase 0.5-0.65: landing contact - keep foot closer to body
        elif phase < 0.65:
            t = (phase - 0.5) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[2] += self.step_height * 0.5 * (1.0 - smooth_t)
            foot[0] += self.step_length * 0.4  # Reduced from 0.7
            foot[2] -= self.compression_depth * 0.2 * smooth_t
        
        # Phase 0.65-0.8: continued compression with all legs
        elif phase < 0.8:
            t = (phase - 0.65) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[0] += self.step_length * 0.4
            foot[2] -= self.compression_depth * (0.2 + 0.3 * smooth_t)
        
        # Phase 0.8-1.0: full compression prep
        else:
            t = (phase - 0.8) / 0.2
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[0] += self.step_length * 0.4
            foot[2] -= self.compression_depth * (0.5 + 0.5 * smooth_t)
        
        return foot

    def _compute_front_leg_trajectory(self, foot_base, phase):
        """
        Front leg trajectory: follows rear legs, provides main thrust.
        Reduced forward extension during landing to prevent joint limits and ground penetration.
        """
        foot = foot_base.copy()
        
        # Phase 0.0-0.15: stance, push base upward and forward
        if phase < 0.15:
            t = phase / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[0] -= self.step_length * 0.25 * smooth_t
            foot[2] -= 0.015 * smooth_t  # Slight compression during push
        
        # Phase 0.15-0.3: liftoff and retraction
        elif phase < 0.3:
            t = (phase - 0.15) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[0] -= self.step_length * 0.25 * (1.0 - smooth_t)
            foot[2] += self.step_height * 0.5 * smooth_t
            foot[2] -= 0.015 * (1.0 - smooth_t)
        
        # Phase 0.3-0.5: flight, extend forward gradually
        elif phase < 0.5:
            t = (phase - 0.3) / 0.2
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[2] += self.step_height * (0.5 + 0.3 * smooth_t)
            foot[0] += self.step_length * 0.3 * smooth_t
        
        # Phase 0.5-0.65: continue extending toward landing
        elif phase < 0.65:
            t = (phase - 0.5) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[2] += self.step_height * 0.8 * (1.0 - 0.4 * smooth_t)
            foot[0] += self.step_length * (0.3 + 0.15 * smooth_t)
        
        # Phase 0.65-0.8: landing contact - keep foot closer to body
        elif phase < 0.8:
            t = (phase - 0.65) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[2] += self.step_height * 0.48 * (1.0 - smooth_t)
            foot[0] += self.step_length * 0.45  # Reduced from 0.8
            foot[2] -= self.compression_depth * 0.3 * smooth_t
        
        # Phase 0.8-1.0: full compression prep
        else:
            t = (phase - 0.8) / 0.2
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[0] += self.step_length * 0.45
            foot[2] -= self.compression_depth * (0.3 + 0.7 * smooth_t)
        
        return foot