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
        
        # Base foot positions in body frame with vertical safety margin
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            pos = v.copy()
            pos[2] += 0.03  # Add vertical offset for ground clearance margin
            self.base_feet_pos_body[k] = pos
        
        # Lunge motion parameters - reduced to respect joint limits
        self.front_lunge_distance = 0.16  # Reduced from 0.25
        self.rear_gather_distance = 0.12  # Reduced from 0.2
        self.rear_drag_distance = 0.08    # Reduced from 0.12
        self.rear_push_distance = 0.10    # Reduced from 0.15
        
        # Base motion parameters
        self.vx_lunge = 0.8      # Forward velocity during lunge phase
        self.vx_recovery = 0.3   # Reduced velocity during recovery
        self.vx_coast = 0.5      # Steady coast velocity
        self.vx_push = 0.9       # Increased velocity during push
        
        # Pitch motion parameters (angular rates) - moderated
        self.pitch_down_rate = -0.5   # Reduced from -0.8
        self.pitch_up_rate = 0.7      # Reduced from 1.0
        self.pitch_preload_rate = -0.2  # Reduced from -0.3
        
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
            vz = 0.0  # Removed downward motion to prevent ground penetration
            
        elif phase < 0.5:
            # Recovery gather phase
            local_phase = (phase - 0.3) / 0.2
            vx = self.vx_lunge * (1.0 - local_phase) + self.vx_recovery * local_phase
            pitch_rate = self.pitch_up_rate
            vz = 0.04 * local_phase  # Slight upward motion as pitch recovers
            
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
            vz = 0.02 * local_phase * (1.0 - local_phase)  # Slight upward then settle
        
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
        - [0.0-0.3]: extend aggressively forward with lift
        - [0.3-0.7]: locked in extended position
        - [0.7-1.0]: retract toward body center with lift
        
        Rear legs (RL, RR):
        - [0.0-0.3]: slight rearward positioning
        - [0.3-0.5]: rapidly gather forward with substantial lift
        - [0.5-0.7]: hold aligned position
        - [0.7-1.0]: extend rearward while pushing with lift
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_name.startswith('F'):
            # Front legs: FL, FR
            if phase < 0.3:
                # Aggressive lunge extension
                local_phase = phase / 0.3
                extension = self.front_lunge_distance * self._smooth_step(local_phase)
                foot[0] += extension
                # Lift during extension to reduce leg flatness and joint stress
                foot[2] += 0.04 * np.sin(np.pi * local_phase)
                # Pitch compensation: lift more when base pitches down
                foot[2] += 0.02 * local_phase
                
            elif phase < 0.7:
                # Locked in extended position
                foot[0] += self.front_lunge_distance
                foot[2] += 0.02  # Maintain slight elevation
                
            else:
                # Retract toward body center
                local_phase = (phase - 0.7) / 0.3
                extension = self.front_lunge_distance * (1.0 - self._smooth_step(local_phase))
                foot[0] += extension
                # Lift during retraction
                foot[2] += 0.05 * np.sin(np.pi * local_phase) + 0.02 * (1.0 - local_phase)
        
        else:
            # Rear legs: RL, RR
            if phase < 0.3:
                # Slight rearward positioning (reduced drag)
                local_phase = phase / 0.3
                drag = -self.rear_drag_distance * local_phase
                foot[0] += drag
                # Maintain elevation during drag
                foot[2] += 0.025
                
            elif phase < 0.5:
                # Rapid gather forward with substantial lift
                local_phase = (phase - 0.3) / 0.2
                start_pos = -self.rear_drag_distance
                end_pos = self.rear_gather_distance
                gather = start_pos + (end_pos - start_pos) * self._smooth_step(local_phase)
                foot[0] += gather
                # Substantial lift arc during gathering to clear ground
                lift_height = 0.08 * np.sin(np.pi * local_phase)
                foot[2] += lift_height + 0.04  # Base clearance plus arc
                
            elif phase < 0.7:
                # Hold aligned position
                foot[0] += self.rear_gather_distance
                foot[2] += 0.03  # Maintain elevation
                
            else:
                # Push extension rearward with lift
                local_phase = (phase - 0.7) / 0.3
                forward_pos = self.rear_gather_distance
                push_extension = -self.rear_push_distance * self._smooth_step(local_phase)
                foot[0] += forward_pos + push_extension
                # Lift during push to prevent penetration and joint stress
                foot[2] += 0.06 * np.sin(np.pi * local_phase) + 0.03 * (1.0 - local_phase)
        
        return foot
    
    def _smooth_step(self, t):
        """
        Smooth interpolation function using cubic smoothstep.
        Maps [0,1] -> [0,1] with zero derivatives at endpoints.
        """
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)