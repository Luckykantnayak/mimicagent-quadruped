from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_DIAGONAL_PITCH_WAVE_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal pitch wave locomotion skill.
    
    Robot moves diagonally forward-right with a pitch wave oscillating along
    the diagonal axis (rear-left to front-right). Body rocks through coordinated
    roll and pitch to create effective rotation about ~45° diagonal axis.
    
    All four legs maintain continuous ground contact throughout cycle.
    Propulsion alternates between rear-pair push (phase 0.0-0.3) and 
    front-pair pull (phase 0.6-1.0), with smooth transition (phase 0.3-0.6).
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.6  # Cycle frequency (Hz) - moderate for smooth pitch wave
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Diagonal velocity parameters (1:1 ratio for 45° diagonal)
        self.vx_base = 0.25  # Forward velocity component (m/s)
        self.vy_base = 0.25  # Rightward velocity component (m/s)
        
        # Angular velocity amplitudes for diagonal pitch wave
        self.roll_amp = 0.15  # rad/s - right/left tilt component
        self.pitch_amp = 0.15  # rad/s - rear/front lift component
        
        # Leg motion parameters (body frame trajectories)
        self.push_extension = 0.06  # Rear leg extension during push (m)
        self.pull_retraction = 0.06  # Front leg retraction during pull (m)
        
        # State variables
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
    def smooth_envelope(self, phase):
        """
        C2-continuous envelope function over full phase cycle [0, 1].
        Uses quintic polynomial to ensure smooth velocity and acceleration.
        """
        # Quintic smoothstep: 6t^5 - 15t^4 + 10t^3
        t = np.clip(phase, 0.0, 1.0)
        return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
    
    def phase_envelope_roll(self, phase):
        """
        C2-continuous roll rate envelope: positive (right tilt) in rear phase,
        negative (left tilt) in front phase, smooth transition through zero.
        """
        if phase < 0.3:
            # Rear phase: ramp up to positive peak
            local_phase = phase / 0.3
            return self.smooth_envelope(local_phase)
        elif phase < 0.7:
            # Transition through zero with smooth reversal
            local_phase = (phase - 0.3) / 0.4
            # Cosine provides C-infinity smoothness
            return np.cos(np.pi * local_phase)
        else:
            # Front phase: ramp down from negative peak to zero
            local_phase = (phase - 0.7) / 0.3
            return -(1.0 - self.smooth_envelope(local_phase))
    
    def phase_envelope_pitch(self, phase):
        """
        C2-continuous pitch rate envelope: negative (rear up) in rear phase,
        positive (front up) in front phase, smooth transition through zero.
        """
        if phase < 0.3:
            # Rear phase: ramp up to negative peak
            local_phase = phase / 0.3
            return -self.smooth_envelope(local_phase)
        elif phase < 0.7:
            # Transition through zero with smooth reversal
            local_phase = (phase - 0.3) / 0.4
            return -np.cos(np.pi * local_phase)
        else:
            # Front phase: ramp down from positive peak to zero
            local_phase = (phase - 0.7) / 0.3
            return (1.0 - self.smooth_envelope(local_phase))
        
    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent diagonal velocity and angular rates.
        Uses C2-continuous envelopes to ensure smooth acceleration profiles.
        """
        
        # Continuous diagonal linear velocity (constant magnitude)
        vx = self.vx_base
        vy = self.vy_base
        vz = 0.0
        
        # C2-continuous angular velocities for diagonal pitch wave
        roll_envelope = self.phase_envelope_roll(phase)
        pitch_envelope = self.phase_envelope_pitch(phase)
        
        roll_rate = self.roll_amp * roll_envelope
        pitch_rate = self.pitch_amp * pitch_envelope
        yaw_rate = 0.0  # Removed yaw coupling to reduce complexity
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        # Integrate pose in world frame
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        
        All legs maintain ground contact. Foot trajectories use C2-continuous
        envelopes to ensure smooth joint velocities and accelerations.
        Vertical adjustments removed to prevent ground penetration conflicts.
        """
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Define smooth fore-aft motion envelopes for rear and front legs
        # Rear legs: RL, RR
        if leg_name.startswith('RL') or leg_name.startswith('RR'):
            
            if phase < 0.35:
                # Active push phase: foot extends backward
                local_phase = phase / 0.35
                envelope = self.smooth_envelope(local_phase)
                extension = self.push_extension * np.sin(np.pi * envelope)
                foot[0] -= extension
                
            elif phase < 0.65:
                # Transition: return to neutral position with C2 continuity
                local_phase = (phase - 0.35) / 0.3
                envelope = self.smooth_envelope(local_phase)
                # Smoothly decay from peak back to zero
                extension = self.push_extension * np.sin(np.pi * (1.0 - envelope))
                foot[0] -= extension
                
            else:
                # Support phase: minimal motion
                pass
        
        # Front legs: FL, FR
        else:
            
            if phase < 0.35:
                # Support phase: minimal motion
                pass
                
            elif phase < 0.65:
                # Preparation: slight forward positioning with C2 continuity
                local_phase = (phase - 0.35) / 0.3
                envelope = self.smooth_envelope(local_phase)
                foot[0] += self.pull_retraction * 0.15 * envelope
                
            else:
                # Active pull phase: foot retracts backward
                local_phase = (phase - 0.65) / 0.35
                envelope = self.smooth_envelope(local_phase)
                retraction = self.pull_retraction * np.sin(np.pi * envelope)
                foot[0] -= retraction
        
        return foot