from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ORBIT_TILT_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Orbit with synchronized roll oscillation.
    
    The robot executes a continuous circular orbit while rhythmically rolling 
    its base left and right. Legs dynamically adjust their lateral stance width:
    outer legs extend, inner legs retract, to support the tilted base throughout 
    the circular path.
    
    - Base motion: constant forward velocity + constant yaw rate (orbit)
                   + sinusoidal roll rate (oscillating tilt)
    - Leg motion: diagonal trot gait with lateral modulation based on roll angle
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Diagonal trot parameters
        self.duty = 0.6
        self.step_length = 0.08
        self.step_height = 0.06
        
        # Base foot positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Diagonal trot phase offsets: FL+RR vs FR+RL
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0
            else:  # FR or RL
                self.phase_offsets[leg] = 0.5
        
        # Orbital motion parameters
        self.vx = 0.3  # forward velocity (m/s)
        self.yaw_rate = 0.5  # constant yaw rate for circular orbit (rad/s)
        
        # Roll oscillation parameters
        self.roll_rate_amp = 1.2  # amplitude of roll rate oscillation (rad/s)
        self.roll_freq = self.freq  # roll oscillates once per cycle
        
        # Lateral extension gain: outer legs extend by this factor times roll angle
        self.lateral_gain = 0.12  # m/rad
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track integrated roll angle for leg modulation
        self.current_roll = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward velocity, constant yaw rate,
        and sinusoidal roll rate.
        
        Roll rate oscillates: negative in [0, 0.25] and [0.5, 0.75],
                              positive in [0.25, 0.5] and [0.75, 1.0]
        This creates left-right-left-right roll pattern synchronized with orbit.
        """
        # Constant forward velocity for orbit
        vx = self.vx
        
        # Sinusoidal roll rate: sin(2*pi*phase) gives correct quadrant behavior
        # phase 0->0.25: sin goes 0->1 but we want negative roll rate (roll left)
        # Use -sin(2*pi*phase) to get desired pattern
        roll_rate = -self.roll_rate_amp * np.sin(2 * np.pi * phase)
        
        # Constant yaw rate for circular orbit
        yaw_rate = self.yaw_rate
        
        # Set velocity commands
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Update current roll angle for leg lateral modulation
        roll, pitch, yaw = quat_to_euler(self.root_quat)
        self.current_roll = roll

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with:
        1. Diagonal trot gait pattern (forward/vertical motion)
        2. Lateral modulation based on current roll angle
        
        Outer legs (opposite to roll direction) extend outward.
        Inner legs (same side as roll) retract inward.
        """
        # Get leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is on left or right side
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compute lateral modulation based on roll angle
        # Positive roll = tilted right, so left legs are outer (extend), right legs inner (retract)
        # Negative roll = tilted left, so right legs are outer (extend), left legs inner (retract)
        if is_left_leg:
            # Left legs: positive y is outward
            # When roll > 0 (right tilt), left legs are outer -> increase y
            # When roll < 0 (left tilt), left legs are inner -> decrease y
            lateral_offset = self.lateral_gain * self.current_roll
        else:
            # Right legs: negative y is outward
            # When roll > 0 (right tilt), right legs are inner -> y less negative
            # When roll < 0 (left tilt), right legs are outer -> y more negative
            lateral_offset = -self.lateral_gain * self.current_roll
        
        foot[1] += lateral_offset
        
        # Diagonal trot gait: stance and swing phases
        if leg_phase < self.duty:
            # Stance phase: foot moves backward relative to body
            progress = leg_phase / self.duty
            foot[0] -= self.step_length * (progress - 0.5)
        else:
            # Swing phase: foot lifts, moves forward with arc
            progress = (leg_phase - self.duty) / (1.0 - self.duty)
            foot[0] += self.step_length * (progress - 0.5)
            
            # Swing height: smooth arc
            swing_angle = np.pi * progress
            foot[2] += self.step_height * np.sin(swing_angle)
        
        return foot