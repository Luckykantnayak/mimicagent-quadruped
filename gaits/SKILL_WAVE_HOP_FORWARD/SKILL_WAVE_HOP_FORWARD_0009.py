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
        
        # Motion parameters - tuned for joint safety and ground clearance
        self.step_height = 0.09  # Slightly reduced for safer landing
        self.step_length = 0.16  # Maintained from previous iteration
        self.compression_depth = 0.04  # Reduced further for joint safety
        
        # Base velocity parameters
        self.max_forward_vel = 0.9
        self.max_upward_vel = 1.0
        self.max_pitch_rate = 0.6
        
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
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t_local))
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
        
        # Phase 0.3-0.5: full_flight
        elif phase < 0.5:
            t_local = (phase - 0.3) / 0.2
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t_local))
            vx = self.max_forward_vel * 0.8 * (1.0 - 0.2 * smooth_t)
            vz = self.max_upward_vel * 0.8 * (1.0 - 2.0 * smooth_t)
            pitch_rate = 0.0
        
        # Phase 0.5-0.65: rear_landing - gentler deceleration
        elif phase < 0.65:
            t_local = (phase - 0.5) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t_local))
            vx = self.max_forward_vel * 0.6 * (1.0 - 0.5 * smooth_t)
            vz = -self.max_upward_vel * 0.35 * smooth_t  # Gentler landing
            pitch_rate = -self.max_pitch_rate * 0.4 * smooth_t
        
        # Phase 0.65-0.8: front_landing
        elif phase < 0.8:
            t_local = (phase - 0.65) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t_local))
            vx = self.max_forward_vel * 0.3 * (1.0 - smooth_t)
            vz = -self.max_upward_vel * 0.15 * smooth_t  # Gentler settling
            pitch_rate = self.max_pitch_rate * 0.25 * np.sin(np.pi * smooth_t)
        
        # Phase 0.8-1.0: compression_prep
        else:
            t_local = (phase - 0.8) / 0.2
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t_local))
            vx = 0.0
            vz = -0.15 * np.sin(np.pi * smooth_t)
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
        Landing trajectories maintain ground clearance until final touchdown.
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
        Soft touchdown approach with maintained clearance during landing phase.
        """
        foot = foot_base.copy()
        
        # Phase 0.0-0.15: rapid liftoff and retraction
        if phase < 0.15:
            t = phase / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[2] += self.step_height * 0.35 * smooth_t
            foot[0] += self.step_length * 0.1 * smooth_t
        
        # Phase 0.15-0.3: continue upward, tucked
        elif phase < 0.3:
            t = (phase - 0.15) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[2] += self.step_height * (0.35 + 0.45 * smooth_t)
            foot[0] += self.step_length * (0.1 + 0.15 * smooth_t)
        
        # Phase 0.3-0.5: apex and begin extending for landing
        elif phase < 0.5:
            t = (phase - 0.3) / 0.2
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[2] += self.step_height * (0.8 - 0.15 * smooth_t)
            foot[0] += self.step_length * (0.25 + 0.15 * smooth_t)
        
        # Phase 0.5-0.65: soft landing - maintain clearance until late in phase
        elif phase < 0.65:
            t = (phase - 0.5) / 0.15
            # Asymptotic descent: stays high until t > 0.6, then drops quickly
            if t < 0.6:
                descent_factor = 0.1 * t / 0.6  # Descend only 10% in first 60% of phase
            else:
                local_t = (t - 0.6) / 0.4
                descent_factor = 0.1 + 0.9 * (0.5 * (1.0 - np.cos(np.pi * local_t)))
            
            foot[2] += self.step_height * 0.65 * (1.0 - descent_factor)
            foot[0] += self.step_length * 0.38  # Reduced from 0.4 for safer joint angles
        
        # Phase 0.65-0.8: ground contact established, no compression yet
        elif phase < 0.8:
            t = (phase - 0.65) / 0.15
            foot[0] += self.step_length * 0.38
            # Foot remains at ground level, no additional descent
        
        # Phase 0.8-1.0: compression - foot moves UP in body frame as base lowers
        else:
            t = (phase - 0.8) / 0.2
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[0] += self.step_length * 0.38
            # Compression: foot rises relative to body as base descends
            foot[2] += self.compression_depth * smooth_t
        
        return foot

    def _compute_front_leg_trajectory(self, foot_base, phase):
        """
        Front leg trajectory: follows rear legs, provides main thrust.
        Reduced forward extension and soft touchdown to prevent joint violations and penetration.
        """
        foot = foot_base.copy()
        
        # Phase 0.0-0.15: stance, push base upward and forward
        if phase < 0.15:
            t = phase / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[0] -= self.step_length * 0.2 * smooth_t
            foot[2] -= 0.01 * smooth_t  # Minimal compression during push
        
        # Phase 0.15-0.3: liftoff and retraction
        elif phase < 0.3:
            t = (phase - 0.15) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[0] -= self.step_length * 0.2 * (1.0 - smooth_t)
            foot[2] -= 0.01 * (1.0 - smooth_t)
            foot[2] += self.step_height * 0.5 * smooth_t
        
        # Phase 0.3-0.5: flight, extend forward moderately
        elif phase < 0.5:
            t = (phase - 0.3) / 0.2
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[2] += self.step_height * (0.5 + 0.25 * smooth_t)
            foot[0] += self.step_length * 0.25 * smooth_t
        
        # Phase 0.5-0.65: continue extending, maintaining height
        elif phase < 0.65:
            t = (phase - 0.5) / 0.15
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[2] += self.step_height * 0.75  # Maintain clearance
            foot[0] += self.step_length * (0.25 + 0.1 * smooth_t)
        
        # Phase 0.65-0.8: soft landing - maintain clearance until late in phase
        elif phase < 0.8:
            t = (phase - 0.65) / 0.15
            # Asymptotic descent similar to rear legs
            if t < 0.6:
                descent_factor = 0.1 * t / 0.6
            else:
                local_t = (t - 0.6) / 0.4
                descent_factor = 0.1 + 0.9 * (0.5 * (1.0 - np.cos(np.pi * local_t)))
            
            foot[2] += self.step_height * 0.75 * (1.0 - descent_factor)
            foot[0] += self.step_length * 0.35  # Reduced from 0.45 to prevent hip joint violations
        
        # Phase 0.8-1.0: compression - foot moves UP in body frame as base lowers
        else:
            t = (phase - 0.8) / 0.2
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            foot[0] += self.step_length * 0.35
            # Compression: foot rises relative to body as base descends
            foot[2] += self.compression_depth * smooth_t
        
        return foot