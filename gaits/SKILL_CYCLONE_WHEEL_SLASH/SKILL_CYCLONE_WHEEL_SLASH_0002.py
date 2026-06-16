from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CYCLONE_WHEEL_SLASH_MotionGenerator(BaseMotionGenerator):
    """
    Cyclone Wheel Slash: High-speed spinning maneuver with controlled lateral slashing.
    
    Motion structure:
    - Phase [0.0, 0.25]: Spin acceleration - build yaw velocity
    - Phase [0.25, 0.5]: Peak spin slash - max yaw rate with rightward lateral drift and leg retraction
    - Phase [0.5, 0.65]: Spin transition - restore traction, decelerate yaw
    - Phase [0.65, 0.85]: Counter slash - leftward lateral acceleration with continued yaw
    - Phase [0.85, 1.0]: Reset and wind-up - return to neutral
    
    All four wheels maintain ground contact throughout. Leg extension modulates
    contact pressure to enable controlled drift during peak spin phases.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Complete cycle in ~1.25 seconds
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.peak_yaw_rate = 4.5  # rad/s - aggressive spin
        self.lateral_slash_velocity = 0.6  # m/s - rightward during peak spin
        self.counter_slash_velocity = 0.65  # m/s - leftward during counter
        self.leg_retraction_amount = 0.035  # m - subtle height reduction during peak spin
        self.forward_burst_amp = 0.3  # m/s - small forward/backward perturbations
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def compute_phase(self, t):
        """Standard phase computation."""
        return (self.freq * t) % 1.0

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Implements spinning with lateral slashing motion.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase [0.0, 0.25]: Spin acceleration
        if phase < 0.25:
            progress = phase / 0.25
            # Smooth ramp-up of yaw rate
            yaw_rate = self.peak_yaw_rate * np.sin(np.pi * 0.5 * progress)
            vx = 0.0
            vy = 0.0
            vz = 0.0
        
        # Phase [0.25, 0.5]: Peak spin slash
        elif phase < 0.5:
            progress = (phase - 0.25) / 0.25
            # Sustained peak yaw rate
            yaw_rate = self.peak_yaw_rate
            # Lateral slash to the right (positive y)
            vy = self.lateral_slash_velocity * np.sin(np.pi * progress)
            # Small forward/backward oscillation
            vx = self.forward_burst_amp * np.sin(4 * np.pi * progress)
            # Slight downward velocity from leg retraction
            vz = -0.15 * np.sin(np.pi * progress)
        
        # Phase [0.5, 0.65]: Spin transition
        elif phase < 0.65:
            progress = (phase - 0.5) / 0.15
            # Decelerate yaw rate
            yaw_rate = self.peak_yaw_rate * (1.0 - 0.4 * progress)
            # Decelerate lateral motion
            vy = self.lateral_slash_velocity * 0.3 * (1.0 - progress)
            # Upward velocity from leg re-extension
            vz = 0.2 * np.sin(np.pi * progress)
            vx = 0.0
        
        # Phase [0.65, 0.85]: Counter slash
        elif phase < 0.85:
            progress = (phase - 0.65) / 0.2
            # Moderate sustained yaw rate
            yaw_rate = self.peak_yaw_rate * 0.6
            # Lateral counter-slash to the left (negative y)
            vy = -self.counter_slash_velocity * np.sin(np.pi * progress)
            # Sharp forward/backward burst
            vx = self.forward_burst_amp * 1.5 * np.sin(2 * np.pi * progress)
            vz = 0.0
        
        # Phase [0.85, 1.0]: Reset and wind-up
        else:
            progress = (phase - 0.85) / 0.15
            # Return yaw rate smoothly toward initial acceleration value
            yaw_rate = self.peak_yaw_rate * 0.6 * (1.0 - progress)
            # Return lateral velocity to neutral
            vy = -self.counter_slash_velocity * 0.2 * (1.0 - np.sin(np.pi * 0.5 * progress))
            vx = 0.0
            vz = 0.0
        
        # Set world frame velocities
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
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
        Compute foot position in body frame for given leg and phase.
        
        All legs maintain ground contact but modulate extension to control
        contact pressure and enable drift during peak spin.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg type for asymmetric motion during counter-slash
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front_leg = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_diagonal_high = leg_name.startswith('FL') or leg_name.startswith('RR')
        
        # Phase [0.0, 0.25]: Spin acceleration - firm stance
        if phase < 0.25:
            # Maintain nominal stance position
            # Diagonal pairs (FL-RR) slightly forward, (FR-RL) slightly back for yaw torque
            if is_diagonal_high:
                foot[0] += 0.01
            else:
                foot[0] -= 0.01
        
        # Phase [0.25, 0.5]: Peak spin slash - synchronized retraction
        elif phase < 0.5:
            progress = (phase - 0.25) / 0.25
            # All legs retract upward uniformly
            retraction = self.leg_retraction_amount * np.sin(np.pi * progress)
            foot[2] += retraction
            # Slight lateral spreading during retraction for stability
            if is_left_leg:
                foot[1] += 0.015 * progress
            else:
                foot[1] -= 0.015 * progress
        
        # Phase [0.5, 0.65]: Spin transition - re-extension
        elif phase < 0.65:
            progress = (phase - 0.5) / 0.15
            # Legs extend back down to nominal height
            retraction = self.leg_retraction_amount * (1.0 - progress)
            foot[2] += retraction
            # Return lateral spread to neutral
            lateral_offset = 0.015 * (1.0 - progress)
            if is_left_leg:
                foot[1] += lateral_offset
            else:
                foot[1] -= lateral_offset
        
        # Phase [0.65, 0.85]: Counter slash - asymmetric stance modulation
        elif phase < 0.85:
            progress = (phase - 0.65) / 0.2
            # Left legs push outward, right legs push inward to generate leftward force
            if is_left_leg:
                foot[1] += 0.025 * np.sin(np.pi * progress)
            else:
                foot[1] -= 0.025 * np.sin(np.pi * progress)
            # Front legs slightly forward, rear legs slightly back
            if is_front_leg:
                foot[0] += 0.02 * np.sin(np.pi * progress)
            else:
                foot[0] -= 0.02 * np.sin(np.pi * progress)
        
        # Phase [0.85, 1.0]: Reset and wind-up - return to nominal
        else:
            progress = (phase - 0.85) / 0.15
            # Smooth return to base position
            # Any remaining offsets decay
            decay = 1.0 - progress
            if is_left_leg:
                foot[1] += 0.025 * decay * 0.2
            else:
                foot[1] -= 0.025 * decay * 0.2
        
        return foot