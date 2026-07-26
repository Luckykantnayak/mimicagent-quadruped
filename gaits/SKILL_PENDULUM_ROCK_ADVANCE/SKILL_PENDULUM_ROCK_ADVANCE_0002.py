from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_PENDULUM_ROCK_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Pendulum rocking gait for forward locomotion.
    
    The robot advances forward through rhythmic side-to-side rocking:
    - Base rolls left [0.0-0.2]: right legs sweep forward
    - Transition to center [0.2-0.4]: all legs settle
    - Base rolls right [0.4-0.6]: left legs sweep forward
    - Transition to center [0.6-0.8]: all legs settle
    - Preparation [0.8-1.0]: stabilize upright
    
    All four feet maintain ground contact throughout.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slow cycle for controlled rocking
        
        # Base foot positions (BODY frame) with increased ground clearance
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            foot_pos = v.copy()
            # Ensure adequate z-offset for roll compensation
            if foot_pos[2] > -0.05:
                foot_pos[2] = -0.05
            self.base_feet_pos_body[k] = foot_pos
        
        # Motion parameters
        self.forward_velocity = 0.15  # Modest forward speed
        self.roll_amplitude = 0.5  # Reduced roll rate for smaller angles
        self.sweep_distance = 0.12  # Forward sweep distance per leg
        self.ground_clearance = 0.03  # Increased clearance during sweep
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with constant forward velocity and sinusoidal roll rate.
        
        Roll rate pattern:
        - [0.0-0.2]: negative (rolling left)
        - [0.2-0.4]: positive (returning to center)
        - [0.4-0.6]: positive (rolling right)
        - [0.6-0.8]: negative (returning to center)
        - [0.8-1.0]: near zero (stabilizing)
        """
        
        # Constant forward velocity
        vx = self.forward_velocity
        
        # Phase-dependent roll rate
        if phase < 0.2:
            # Rock left
            roll_rate = -self.roll_amplitude * np.sin(np.pi * phase / 0.2)
        elif phase < 0.4:
            # Transition to center from left
            local_phase = (phase - 0.2) / 0.2
            roll_rate = self.roll_amplitude * np.sin(np.pi * local_phase)
        elif phase < 0.6:
            # Rock right
            local_phase = (phase - 0.4) / 0.2
            roll_rate = self.roll_amplitude * np.sin(np.pi * local_phase)
        elif phase < 0.8:
            # Transition to center from right
            local_phase = (phase - 0.6) / 0.2
            roll_rate = -self.roll_amplitude * np.sin(np.pi * local_phase)
        else:
            # Stabilize upright
            local_phase = (phase - 0.8) / 0.2
            roll_rate = -0.1 * self.roll_amplitude * np.sin(np.pi * local_phase)
        
        # Set velocity commands
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def get_roll_angle_from_quat(self, quat):
        """Extract roll angle from quaternion."""
        w, x, y, z = quat
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        return roll

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase.
        
        Left legs (FL, RL) sweep forward during right rock [0.4-0.6]
        Right legs (FR, RR) sweep forward during left rock [0.0-0.2]
        All feet remain grounded with roll-compensation to prevent penetration.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Get current roll angle for compensation
        roll_angle = self.get_roll_angle_from_quat(self.root_quat)
        
        # Lateral distance from centerline (approximate)
        lateral_distance = abs(foot[1])
        
        # Roll compensation: when body rolls, adjust z to maintain ground contact
        # Downward side needs more negative z to prevent penetration
        if is_right_leg:
            # Right legs: compensate for negative (left) roll
            # When roll_angle < 0, right side is up, no extra clearance needed
            # When roll_angle > 0, right side is down, needs compensation
            z_compensation = -lateral_distance * max(0, roll_angle)
        else:
            # Left legs: compensate for positive (right) roll
            # When roll_angle > 0, left side is up, no extra clearance needed
            # When roll_angle < 0, left side is down, needs compensation
            z_compensation = -lateral_distance * max(0, -roll_angle)
        
        foot[2] += z_compensation
        
        if is_right_leg:
            # Right legs sweep during left rock [0.0-0.2]
            if phase < 0.2:
                # Forward sweep with minimal lift
                progress = phase / 0.2
                foot[0] += self.sweep_distance * progress
                foot[2] += self.ground_clearance * np.sin(np.pi * progress)
            elif phase < 0.4:
                # Settling after sweep
                foot[0] += self.sweep_distance
            elif phase < 0.6:
                # Bearing weight during right rock - stationary
                foot[0] += self.sweep_distance
            elif phase < 0.8:
                # Transition - hold position
                foot[0] += self.sweep_distance
            else:
                # Preparation - reset for next cycle (gradual return)
                local_phase = (phase - 0.8) / 0.2
                foot[0] += self.sweep_distance * (1.0 - local_phase)
                
        elif is_left_leg:
            # Left legs sweep during right rock [0.4-0.6]
            if phase < 0.2:
                # Bearing weight during left rock - stationary
                pass
            elif phase < 0.4:
                # Transition - hold position
                pass
            elif phase < 0.6:
                # Forward sweep with minimal lift
                local_phase = (phase - 0.4) / 0.2
                foot[0] += self.sweep_distance * local_phase
                foot[2] += self.ground_clearance * np.sin(np.pi * local_phase)
            elif phase < 0.8:
                # Settling after sweep
                foot[0] += self.sweep_distance
            else:
                # Preparation - reset for next cycle (gradual return)
                local_phase = (phase - 0.8) / 0.2
                foot[0] += self.sweep_distance * (1.0 - local_phase)
        
        return foot