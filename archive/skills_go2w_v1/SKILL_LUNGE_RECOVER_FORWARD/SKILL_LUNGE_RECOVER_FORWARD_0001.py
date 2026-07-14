from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_LUNGE_RECOVER_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Dynamic forward lunge-recover skill with aggressive front-leg extension
    and rear-leg gathering, creating wave-like forward progression.
    
    Phase structure:
    - [0.0-0.3] aggressive_lunge: front legs extend forward, base pitches down
    - [0.3-0.5] recovery_gather: rear legs gather forward, base pitches up
    - [0.5-0.7] aligned_coast: all legs aligned, neutral pitch, steady forward
    - [0.7-1.0] push_preload: rear legs push, front legs retract, prepare next cycle
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.6  # Slower frequency for dynamic lunge motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Lunge motion parameters
        self.front_lunge_distance = 0.25  # Aggressive forward extension
        self.rear_gather_distance = 0.2   # Rapid forward gathering
        self.rear_push_distance = 0.15    # Rear leg push extension
        
        # Base motion parameters
        self.vx_lunge = 0.8      # Forward velocity during lunge phase
        self.vx_recovery = 0.3   # Reduced velocity during recovery
        self.vx_coast = 0.5      # Steady coast velocity
        self.vx_push = 0.9       # Increased velocity during push
        
        # Pitch motion parameters (angular rates)
        self.pitch_down_rate = -0.8   # Nose-down during lunge
        self.pitch_up_rate = 1.0      # Nose-up during recovery
        self.pitch_preload_rate = -0.3  # Slight nose-down for next cycle
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity and angular velocity based on phase.
        Phase-dependent control:
        - [0.0-0.3]: lunge forward with nose-down pitch
        - [0.3-0.5]: recovery with nose-up pitch
        - [0.5-0.7]: coast with neutral pitch
        - [0.7-1.0]: push with slight nose-down preparation
        """
        
        # Determine current phase range
        if phase < 0.3:
            # Aggressive lunge phase
            local_phase = phase / 0.3
            vx = self.vx_lunge * (0.6 + 0.4 * local_phase)  # Increasing velocity
            pitch_rate = self.pitch_down_rate
            vz = -0.05 * local_phase  # Slight downward motion initially
            
        elif phase < 0.5:
            # Recovery gather phase
            local_phase = (phase - 0.3) / 0.2
            vx = self.vx_lunge * (1.0 - local_phase) + self.vx_recovery * local_phase
            pitch_rate = self.pitch_up_rate
            vz = 0.08 * local_phase  # Upward motion as pitch recovers
            
        elif phase < 0.7:
            # Aligned coast phase
            local_phase = (phase - 0.5) / 0.2
            vx = self.vx_coast
            pitch_rate = 0.0
            vz = 0.0
            
        else:
            # Push preload phase
            local_phase = (phase - 0.7) / 0.3
            vx = self.vx_coast * (1.0 - local_phase) + self.vx_push * local_phase
            pitch_rate = self.pitch_preload_rate * local_phase
            vz = 0.03 * local_phase * (1.0 - local_phase)  # Slight upward then settle
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, 0.0, vz])
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
        Compute foot position in body frame based on phase and leg.
        
        Front legs (FL, FR):
        - [0.0-0.3]: extend aggressively forward
        - [0.3-0.7]: locked in extended position
        - [0.7-1.0]: retract toward body center
        
        Rear legs (RL, RR):
        - [0.0-0.3]: drag passively rearward
        - [0.3-0.5]: rapidly gather forward
        - [0.5-0.7]: hold aligned position
        - [0.7-1.0]: extend rearward while pushing
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_name.startswith('F'):
            # Front legs: FL, FR
            if phase < 0.3:
                # Aggressive lunge extension
                local_phase = phase / 0.3
                extension = self.front_lunge_distance * self._smooth_step(local_phase)
                foot[0] += extension
                # Slight height variation during extension
                foot[2] += 0.02 * np.sin(np.pi * local_phase)
                
            elif phase < 0.7:
                # Locked in extended position
                foot[0] += self.front_lunge_distance
                
            else:
                # Retract toward body center
                local_phase = (phase - 0.7) / 0.3
                extension = self.front_lunge_distance * (1.0 - self._smooth_step(local_phase))
                foot[0] += extension
                # Slight height variation during retraction
                foot[2] += 0.015 * np.sin(np.pi * local_phase)
        
        else:
            # Rear legs: RL, RR
            if phase < 0.3:
                # Passive drag rearward
                local_phase = phase / 0.3
                drag = -0.12 * local_phase
                foot[0] += drag
                
            elif phase < 0.5:
                # Rapid gather forward
                local_phase = (phase - 0.3) / 0.2
                start_pos = -0.12
                end_pos = self.rear_gather_distance
                gather = start_pos + (end_pos - start_pos) * self._smooth_step(local_phase)
                foot[0] += gather
                # Slight lift during rapid gathering
                foot[2] += 0.03 * np.sin(np.pi * local_phase)
                
            elif phase < 0.7:
                # Hold aligned position
                foot[0] += self.rear_gather_distance
                
            else:
                # Push extension rearward
                local_phase = (phase - 0.7) / 0.3
                forward_pos = self.rear_gather_distance
                push_extension = -self.rear_push_distance * self._smooth_step(local_phase)
                foot[0] += forward_pos + push_extension
        
        return foot
    
    def _smooth_step(self, t):
        """
        Smooth interpolation function using cubic smoothstep.
        Maps [0,1] -> [0,1] with zero derivatives at endpoints.
        """
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)