from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ORBIT_TILT_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Circular locomotion with synchronized rhythmic roll oscillations.
    
    - Robot executes circular orbit in horizontal plane
    - Base rolls left and right in coordination with orbital position
    - Diagonal leg pairs alternate stance/swing (trot-like gait)
    - Outer legs extend more during roll to maintain ground contact
    - Two complete roll cycles per orbit cycle
    """

    def __init__(self, base_init_feet_pos):
        # Initialize base class
        BaseMotionGenerator.__init__(self, base_init_feet_pos, freq=0.5)
        
        # Gait parameters
        self.duty = 0.5  # 50% duty cycle for trot-like diagonal gait
        self.step_height = 0.10  # Moderate swing height for clearance
        
        # Orbital motion parameters
        self.orbit_radius = 1.5  # Radius of circular path
        self.orbit_angular_velocity = 2 * np.pi * self.freq  # Complete circle per cycle
        
        # Roll oscillation parameters
        self.roll_amplitude = 0.35  # Roll angle amplitude (radians, ~20 degrees)
        self.roll_frequency = 2.0 * self.freq  # Two roll cycles per orbit
        
        # Linear velocity magnitude for circular motion
        self.linear_speed = self.orbit_radius * self.orbit_angular_velocity
        
        # Leg extension modulation for roll compensation
        self.extension_amplitude = 0.06  # Radial extension variation
        
        # Phase offsets for diagonal pairs (Group 1: FL/RR at 0, Group 2: FR/RL at 0.5)
        self.phase_offsets = {}
        for leg_name in self.leg_names:
            if leg_name.startswith('FL') or leg_name.startswith('RR'):
                self.phase_offsets[leg_name] = 0.0  # Group 1
            else:  # FR or RL
                self.phase_offsets[leg_name] = 0.5  # Group 2

    def update_base_motion(self, phase, dt):
        """
        Update base pose with circular trajectory and synchronized roll oscillation.
        
        Linear velocity creates circular motion in horizontal plane.
        Roll rate oscillates sinusoidally (two cycles per orbit).
        Yaw rate maintains circular trajectory.
        """
        # Circular trajectory: parametric circle in world frame
        # vx = v * cos(theta), vy = v * sin(theta), where theta = 2*pi*phase
        orbit_angle = 2 * np.pi * phase
        vx = self.linear_speed * np.cos(orbit_angle)
        vy = self.linear_speed * np.sin(orbit_angle)
        vz = 0.0
        
        # Roll oscillation: two cycles per orbit (4*pi*phase)
        # Phase 0-0.25: roll left (negative)
        # Phase 0.25-0.5: roll right (positive)
        # Phase 0.5-0.75: roll left (negative)
        # Phase 0.75-1.0: return to neutral
        roll_phase = 4 * np.pi * phase
        roll_rate = -self.roll_amplitude * 4 * np.pi * self.freq * np.cos(roll_phase)
        
        pitch_rate = 0.0
        
        # Constant yaw rate for circular trajectory
        yaw_rate = self.orbit_angular_velocity
        
        # Set velocity commands
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
        Compute foot trajectory in body frame.
        
        - Stance phase: foot modulates radial extension based on roll angle
          - Outer legs (relative to roll direction) extend more
          - Inner legs retract slightly
        - Swing phase: smooth arc with forward advancement and moderate clearance
        """
        # Get leg-specific phase with diagonal pair offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Base foot position
        foot = self.base_feet_pos[leg_name].copy()
        
        # Determine if leg is in Group 1 (FL/RR) or Group 2 (FR/RL)
        is_group_1 = leg_name.startswith('FL') or leg_name.startswith('RR')
        
        # Determine current roll state from global phase
        # Roll oscillates: negative (left) at phase ~0.125 and ~0.625, positive (right) at ~0.375
        roll_phase = 4 * np.pi * phase
        current_roll_angle = self.roll_amplitude * np.sin(roll_phase)
        
        if leg_phase < self.duty:
            # ============ STANCE PHASE ============
            # Modulate radial extension based on roll angle and leg position
            
            stance_progress = leg_phase / self.duty
            
            # Determine if this leg is on left or right side
            is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
            is_front_leg = leg_name.startswith('FL') or leg_name.startswith('FR')
            
            # Extension logic:
            # When rolling left (negative roll), left legs extend more (outer), right legs retract (inner)
            # When rolling right (positive roll), right legs extend more (outer), left legs retract (inner)
            if is_left_leg:
                # Left legs: extend when roll is negative (left roll), retract when positive (right roll)
                extension_factor = -current_roll_angle / self.roll_amplitude
            else:
                # Right legs: extend when roll is positive (right roll), retract when negative (left roll)
                extension_factor = current_roll_angle / self.roll_amplitude
            
            # Apply radial extension in x-y plane (body frame)
            radial_extension = extension_factor * self.extension_amplitude
            
            # Extend outward from body center
            lateral_sign = 1.0 if is_left_leg else -1.0
            foot[1] += lateral_sign * radial_extension * 0.7  # Lateral component
            
            # Also slight forward/back modulation for front/rear legs
            forward_sign = 1.0 if is_front_leg else -1.0
            foot[0] += forward_sign * radial_extension * 0.3  # Longitudinal component
            
        else:
            # ============ SWING PHASE ============
            swing_progress = (leg_phase - self.duty) / (1.0 - self.duty)
            
            # Forward advancement during swing (body frame x-direction)
            step_length = 0.12
            foot[0] += step_length * (swing_progress - 0.5)
            
            # Vertical clearance with smooth arc
            swing_angle = np.pi * swing_progress
            foot[2] += self.step_height * np.sin(swing_angle)
            
            # Slight lateral adjustment during swing for smoother transition
            is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
            lateral_sign = 1.0 if is_left_leg else -1.0
            lateral_adjustment = 0.02 * np.sin(swing_angle)
            foot[1] += lateral_sign * lateral_adjustment
        
        return foot