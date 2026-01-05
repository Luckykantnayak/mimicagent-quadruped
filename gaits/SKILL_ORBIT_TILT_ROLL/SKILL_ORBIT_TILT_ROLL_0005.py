from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ORBIT_TILT_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Orbital locomotion with continuous roll oscillation.
    
    The robot executes a circular path while rolling left and right in a sinusoidal pattern.
    Uses diagonal gait coordination with asymmetric leg extension based on roll angle.
    
    - Base motion: Constant yaw rate for circular orbit, sinusoidal roll rate
    - Leg motion: Diagonal gait with swing timing coordinated to roll phase
    - Extension asymmetry: Legs on lowered side extend more to maintain contact
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        # Store leg names
        self.leg_names = leg_names
        
        # Motion parameters
        self.freq = 1.0  # One complete orbit + roll cycle per second
        
        # Base velocities for orbital motion
        self.vx_base = 0.3  # Forward velocity in body frame
        self.vy_amplitude = 0.05  # Lateral velocity modulation amplitude
        self.yaw_rate_base = 1.2  # Constant yaw rate for circular orbit
        self.roll_rate_amplitude = 2.0  # Roll oscillation amplitude (rad/s)
        self.roll_freq = 1.0  # Roll oscillates once per orbit cycle
        
        # Gait parameters
        self.step_length = 0.12
        self.step_height = 0.06
        self.lateral_swing_offset = 0.02
        
        # Base foot positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets and swing timing for diagonal gait
        # FL and RR are one diagonal pair, FR and RL are the other
        self.swing_phases = {
            leg_names[0]: [(0.75, 1.0)],  # FL swings in fourth quadrant
            leg_names[1]: [(0.125, 0.25), (0.5, 0.625)],  # FR swings twice
            leg_names[2]: [(0.25, 0.375), (0.625, 0.75)],  # RL swings twice
            leg_names[3]: [(0.875, 1.0)],  # RR swings at end of cycle
        }
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Initialize base class
        BaseMotionGenerator.__init__(self, initial_foot_positions_body, freq=self.freq)

    def update_base_motion(self, phase, dt):
        """
        Update base pose with circular orbital motion and sinusoidal roll oscillation.
        
        Base velocities:
        - vx: constant forward velocity
        - vy: sinusoidal lateral velocity modulated by phase
        - yaw_rate: constant positive for circular motion
        - roll_rate: sinusoidal oscillation, peaks at phase 0.25 (left) and 0.75 (left again)
        """
        # Linear velocity in body frame, then transform to world
        # Lateral velocity alternates to assist orbital curvature
        if phase < 0.25:
            vy = -self.vy_amplitude
        elif phase < 0.5:
            vy = self.vy_amplitude
        elif phase < 0.75:
            vy = -self.vy_amplitude
        else:
            vy = self.vy_amplitude
        
        vx = self.vx_base
        vz = 0.0
        
        # Transform body-frame velocity to world frame
        vel_body = np.array([vx, vy, vz])
        R = quat_to_rotation_matrix(self.root_quat)
        self.vel_world = R @ vel_body
        
        # Angular velocity: constant yaw rate + sinusoidal roll rate
        # Roll rate oscillates: negative in [0, 0.25] and [0.5, 0.75], positive elsewhere
        # This creates leftward roll peaks at 0.25 and 0.75, rightward peak at 0.5, neutral at 0.0 and 1.0
        roll_rate = self.roll_rate_amplitude * np.sin(2 * np.pi * self.roll_freq * phase)
        pitch_rate = 0.0
        yaw_rate = self.yaw_rate_base
        
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
        Compute foot position in body frame with diagonal gait and roll-based extension.
        
        - Stance: foot slides rearward, extension modulated by roll angle
        - Swing: foot lifts with arc, repositions forward
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine current roll angle from quaternion
        roll, _, _ = quat_to_euler(self.root_quat)
        
        # Determine if leg is in swing or stance
        in_swing = False
        swing_progress = 0.0
        
        for swing_start, swing_end in self.swing_phases[leg_name]:
            if swing_start < swing_end:
                if swing_start <= phase < swing_end:
                    in_swing = True
                    swing_progress = (phase - swing_start) / (swing_end - swing_start)
                    break
            else:  # Wraps around phase=1.0
                if phase >= swing_start or phase < swing_end:
                    in_swing = True
                    if phase >= swing_start:
                        swing_progress = (phase - swing_start) / (1.0 + swing_end - swing_start)
                    else:
                        swing_progress = (1.0 - swing_start + phase) / (1.0 + swing_end - swing_start)
                    break
        
        if in_swing:
            # Swing phase: arc trajectory
            # Lift foot, move forward, add lateral offset, then descend
            swing_angle = np.pi * swing_progress
            
            # Forward displacement
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # Vertical arc
            foot[2] += self.step_height * np.sin(swing_angle)
            
            # Lateral offset during swing (inward for left legs, outward for right legs)
            if leg_name.startswith('FL') or leg_name.startswith('RL'):
                foot[1] += self.lateral_swing_offset * np.sin(swing_angle)
            else:
                foot[1] -= self.lateral_swing_offset * np.sin(swing_angle)
        else:
            # Stance phase: rearward slide with roll-based extension modulation
            # Calculate stance progress (inverse of phase within stance range)
            # For simplicity, use phase-based rearward displacement
            foot[0] -= self.step_length * 0.3 * np.sin(2 * np.pi * phase)
            
            # Extension modulation based on roll:
            # - Negative roll (left tilt): left legs elevated, right legs lowered
            # - Positive roll (right tilt): right legs elevated, left legs lowered
            # Legs on lowered side extend more (more negative z)
            
            if leg_name.startswith('FL') or leg_name.startswith('RL'):
                # Left side legs
                # If roll < 0 (left tilt), these are elevated -> reduce extension
                # If roll > 0 (right tilt), these are lowered -> increase extension
                extension_adjustment = 0.04 * roll
            else:
                # Right side legs
                # If roll < 0 (left tilt), these are lowered -> increase extension
                # If roll > 0 (right tilt), these are elevated -> reduce extension
                extension_adjustment = -0.04 * roll
            
            foot[2] += extension_adjustment
        
        return foot