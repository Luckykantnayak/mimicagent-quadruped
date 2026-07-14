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
        # Roll and pitch combine to create rotation about diagonal axis
        self.roll_amp = 0.15  # rad/s - right/left tilt component
        self.pitch_amp = 0.15  # rad/s - rear/front lift component
        
        # Leg motion parameters (body frame trajectories)
        self.push_extension = 0.06  # Rear leg extension during push (m)
        self.pull_retraction = 0.06  # Front leg retraction during pull (m)
        self.vertical_adjust = 0.03  # Vertical adjustment during pitch (m)
        
        # State variables
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent diagonal velocity and angular rates.
        
        Phase structure:
        - [0.0, 0.3]: Rear lift/push - negative pitch rate, positive roll rate
        - [0.3, 0.6]: Level transition - rates decay to zero
        - [0.6, 1.0]: Front lift/pull - positive pitch rate, negative roll rate
        
        Roll and pitch rates are synchronized to create diagonal axis rotation.
        """
        
        # Continuous diagonal linear velocity (constant magnitude, consistent direction)
        vx = self.vx_base
        vy = self.vy_base
        vz = 0.0
        
        # Phase-dependent angular velocities for diagonal pitch wave
        if phase < 0.3:
            # Rear lift phase: rear up, slight right tilt
            phase_local = phase / 0.3
            # Smooth ramp up from zero at phase 0
            envelope = np.sin(np.pi * phase_local)
            roll_rate = self.roll_amp * envelope
            pitch_rate = -self.pitch_amp * envelope
            yaw_rate = 0.02 * envelope  # Slight yaw coupling
            
        elif phase < 0.6:
            # Level transition: smoothly return to zero rates
            phase_local = (phase - 0.3) / 0.3
            # Smooth transition through zero
            envelope = np.cos(np.pi * phase_local)
            roll_rate = self.roll_amp * envelope * 0.5
            pitch_rate = -self.pitch_amp * envelope * 0.5
            yaw_rate = 0.01 * envelope
            
        else:
            # Front lift phase: front up, slight left tilt
            phase_local = (phase - 0.6) / 0.4
            # Smooth ramp up then down to return to zero at phase 1.0
            envelope = np.sin(np.pi * phase_local)
            roll_rate = -self.roll_amp * envelope
            pitch_rate = self.pitch_amp * envelope
            yaw_rate = 0.02 * envelope
        
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
        
        All legs maintain ground contact. Body motion creates apparent foot
        motion in body frame. Rear legs extend/retract during push phase,
        front legs retract/extend during pull phase.
        """
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Rear legs: RL, RR
        if leg_name.startswith('RL') or leg_name.startswith('RR'):
            
            if phase < 0.3:
                # Active push phase: foot extends backward and upward
                phase_local = phase / 0.3
                # Smooth extension profile
                extension = self.push_extension * np.sin(np.pi * phase_local)
                foot[0] -= extension  # Backward in body frame
                foot[2] += self.vertical_adjust * np.sin(np.pi * phase_local)  # Upward as rear lifts
                
                # Add slight lateral component for RR to emphasize rightward push
                if leg_name.startswith('RR'):
                    foot[1] += 0.015 * np.sin(np.pi * phase_local)
                    
            elif phase < 0.6:
                # Transition: return to neutral position
                phase_local = (phase - 0.3) / 0.3
                # Smooth forward repositioning
                extension = self.push_extension * (1.0 - phase_local)
                foot[0] -= extension
                foot[2] += self.vertical_adjust * np.cos(np.pi * (phase_local + 0.5))
                
                if leg_name.startswith('RR'):
                    foot[1] += 0.015 * (1.0 - phase_local)
                    
            else:
                # Support phase: minimal motion, slight vertical adjustment for front lift
                phase_local = (phase - 0.6) / 0.4
                # Small downward motion as front lifts (rear lowers relatively)
                foot[2] -= self.vertical_adjust * 0.3 * np.sin(np.pi * phase_local)
        
        # Front legs: FL, FR
        else:  # FL or FR
            
            if phase < 0.3:
                # Support phase: minimal motion, slight vertical adjustment for rear lift
                phase_local = phase / 0.3
                # Small downward motion as rear lifts (front lowers relatively)
                foot[2] -= self.vertical_adjust * 0.3 * np.sin(np.pi * phase_local)
                
            elif phase < 0.6:
                # Transition: prepare for pulling
                phase_local = (phase - 0.3) / 0.3
                # Slight forward positioning
                foot[0] += self.pull_retraction * 0.2 * phase_local
                
                # FR emphasizes rightward positioning
                if leg_name.startswith('FR'):
                    foot[1] += 0.015 * phase_local
                    
            else:
                # Active pull phase: foot retracts backward (body moves forward over it)
                phase_local = (phase - 0.6) / 0.4
                # Smooth retraction profile
                retraction = self.pull_retraction * np.sin(np.pi * phase_local)
                foot[0] -= retraction  # Backward in body frame as body advances
                foot[2] += self.vertical_adjust * np.sin(np.pi * phase_local)  # Upward as front lifts
                
                # FR includes lateral component
                if leg_name.startswith('FR'):
                    foot[1] -= 0.015 * np.sin(np.pi * phase_local)  # Leftward relative motion
                else:  # FL
                    # Slight adjustment for symmetric diagonal motion
                    foot[1] += 0.008 * np.sin(np.pi * phase_local)
        
        return foot