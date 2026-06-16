from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_DIAGONAL_PITCH_WAVE_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal pitch wave locomotion with forward-right travel.
    
    Motion characteristics:
    - Pitch wave propagates rear-left to front-right
    - Diagonal coordination: (RL+FR) vs (RR+FL)
    - Continuous ground contact with brief dual-leg swings
    - Base moves diagonally forward-right with coordinated pitch oscillation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Foot motion parameters
        self.step_length = 0.12  # Forward swing distance
        self.step_width = 0.08   # Lateral adjustment for diagonal motion
        self.step_height = 0.06  # Swing clearance height
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Diagonal velocity components (equal for 45° diagonal)
        self.vx_magnitude = 0.25  # Forward velocity component (m/s)
        self.vy_magnitude = 0.25  # Rightward velocity component (m/s)
        
        # Pitch wave parameters
        self.pitch_amplitude = 0.4  # Max pitch rate (rad/s)
        self.roll_amplitude = 0.15  # Subtle roll for diagonal rocking
        self.yaw_amplitude = 0.1    # Heading correction rate
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base using diagonal velocity and pitch wave propagation.
        
        Phase structure:
        [0.0-0.3]: Rear push, negative pitch rate (rear lifts)
        [0.3-0.6]: Center transition, body levels
        [0.6-1.0]: Front pull, positive pitch rate (front lifts)
        """
        
        # Linear velocity: sustained diagonal forward-right
        vx = self.vx_magnitude
        vy = self.vy_magnitude
        
        # Vertical velocity: slight oscillation as body pitches
        if phase < 0.3:
            # Rear lifts, CoM drops slightly
            vz = -0.02
        elif phase < 0.6:
            # Body levels
            vz = 0.0
        else:
            # Front lifts, CoM rises slightly
            vz = 0.02
        
        # Angular velocity: pitch wave with diagonal coupling
        
        # Pitch rate: rear-up → level → front-up
        if phase < 0.3:
            # Rear push phase: negative pitch (nose down relative to rear)
            pitch_progress = phase / 0.3
            pitch_rate = -self.pitch_amplitude * np.cos(np.pi * pitch_progress)
        elif phase < 0.6:
            # Center transition: pitch rate near zero
            pitch_rate = 0.0
        else:
            # Front pull phase: positive pitch (nose up)
            pitch_progress = (phase - 0.6) / 0.4
            pitch_rate = self.pitch_amplitude * np.cos(np.pi * pitch_progress)
        
        # Roll rate: subtle diagonal rocking
        # Positive during rear push (right side rises), negative during front pull
        if phase < 0.3:
            roll_rate = self.roll_amplitude * np.sin(np.pi * phase / 0.3)
        elif phase < 0.6:
            roll_rate = 0.0
        else:
            roll_rate = -self.roll_amplitude * np.sin(np.pi * (phase - 0.6) / 0.4)
        
        # Yaw rate: maintain diagonal heading
        # Slight positive during rear push, slight negative during front pull
        if phase < 0.3:
            yaw_rate = self.yaw_amplitude * 0.5
        elif phase < 0.6:
            yaw_rate = 0.0
        else:
            yaw_rate = -self.yaw_amplitude * 0.3
        
        # Apply velocities
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
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
        Compute foot trajectory in BODY frame based on phase and leg coordination.
        
        Contact schedule:
        FL: stance [0.0-0.8], swing [0.8-1.0]
        FR: stance [0.0-0.8], swing [0.8-1.0]
        RL: stance [0.0-0.5] and [0.6-1.0], swing [0.5-0.6]
        RR: stance [0.0-0.5] and [0.6-1.0], swing [0.5-0.6]
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Front legs (FL, FR): stance until phase 0.8, then swing
        if leg_name.startswith('F'):
            if phase < 0.3:
                # Rear push phase: front legs provide stable forward support
                # Minimal motion, slight backward slide as body advances
                slide_progress = phase / 0.3
                foot[0] -= 0.03 * slide_progress
                
            elif phase < 0.6:
                # Center transition: front legs bear increasing load
                # Continue backward slide in body frame
                slide_progress = (phase - 0.3) / 0.3
                foot[0] -= 0.03 * (1.0 + slide_progress)
                
            elif phase < 0.8:
                # Front pull phase: active drive
                # Maximum backward position, then begin preparing for swing
                pull_progress = (phase - 0.6) / 0.2
                foot[0] -= 0.06 * (1.0 - 0.3 * pull_progress)
                
            else:
                # Swing phase [0.8-1.0]: arc trajectory forward
                swing_progress = (phase - 0.8) / 0.2
                
                # Forward swing
                foot[0] += self.step_length * (swing_progress - 0.5)
                
                # Lateral adjustment for diagonal motion
                if leg_name.startswith('FL'):
                    foot[1] += self.step_width * 0.3 * swing_progress
                else:  # FR
                    foot[1] -= self.step_width * 0.5 * swing_progress
                
                # Arc clearance
                swing_angle = np.pi * swing_progress
                foot[2] += self.step_height * np.sin(swing_angle)
        
        # Rear legs (RL, RR): stance except brief swing [0.5-0.6]
        else:
            if phase < 0.3:
                # Rear push phase: active propulsion
                # Foot extends backward (slides in body frame)
                push_progress = phase / 0.3
                foot[0] -= self.step_length * 0.4 * push_progress
                
            elif phase < 0.5:
                # Transition: unloading
                # Continue backward slide, preparing for swing
                transition_progress = (phase - 0.3) / 0.2
                foot[0] -= self.step_length * (0.4 * (1.0 + transition_progress))
                
            elif phase < 0.6:
                # Swing phase [0.5-0.6]: brief forward repositioning
                swing_progress = (phase - 0.5) / 0.1
                
                # Forward swing to neutral/forward position
                foot[0] += self.step_length * (swing_progress - 0.2)
                
                # Lateral adjustment
                if leg_name.startswith('RL'):
                    foot[1] += self.step_width * 0.4 * swing_progress
                else:  # RR
                    foot[1] -= self.step_width * 0.4 * swing_progress
                
                # Arc clearance (lower than front legs)
                swing_angle = np.pi * swing_progress
                foot[2] += self.step_height * 0.8 * np.sin(swing_angle)
                
            else:
                # Stance phase [0.6-1.0]: rear support during front pull
                # Foot planted, slides backward as body advances
                support_progress = (phase - 0.6) / 0.4
                foot[0] -= 0.04 * support_progress
        
        return foot