from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ELLIPTICAL_LEAN_LOOP_MotionGenerator(BaseMotionGenerator):
    """
    Elliptical path with dynamically varying lean angle.
    
    The robot executes a continuous elliptical trajectory while modulating
    its roll (lean) angle based on path curvature:
    - Maximum inward lean at narrow ends (sharper turns)
    - Minimum lean at wide sides (gentler turns)
    
    Gait: Continuous trot with diagonal leg pairs (FL+RR, FR+RL) alternating.
    Phase cycle: One complete ellipse per phase period [0,1].
    
    Base motion:
    - Forward velocity: constant
    - Yaw rate: sinusoidal, peaks at narrow ends
    - Roll rate: sinusoidal, positive at first narrow end, negative at second
    
    Leg motion:
    - Diagonal pairs alternate every 0.25 phase
    - Stance: foot sweeps rearward in body frame
    - Swing: foot lifts and repositions forward
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Hz, completes one ellipse every 2 seconds
        
        # Trot gait parameters
        self.duty = 0.5  # 50% duty cycle for trot
        self.step_length = 0.12  # meters, fore-aft swing amplitude
        self.step_height = 0.06  # meters, swing height clearance
        
        # Base foot positions (BODY frame reference)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Add vertical margin to base foot positions to accommodate roll-induced height variations
        for leg in leg_names:
            self.base_feet_pos_body[leg][2] += 0.03  # 3cm additional ground clearance
        
        # Phase offsets for diagonal trot coordination
        # FL and RR move together (group_1), FR and RL move together (group_2)
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0  # Group 1
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.phase_offsets[leg] = 0.5  # Group 2, 180 degrees out of phase
        
        # Base motion parameters
        self.vx_forward = 0.4  # m/s, constant forward velocity
        
        # Elliptical path parameters
        # Yaw rate modulation (creates elliptical shape)
        self.yaw_rate_amplitude = 1.2  # rad/s, peak yaw rate at narrow ends
        self.yaw_rate_freq = 1.0  # Cycles per phase (2 peaks per ellipse)
        
        # Roll rate modulation (creates lean variation)
        self.roll_rate_amplitude = 0.8  # rad/s, peak roll rate for lean
        self.roll_rate_freq = 1.0  # Cycles per phase (alternates left/right)
        
        # Lateral foot adjustment during lean
        self.lean_compensation = 0.02  # meters, lateral shift per unit roll angle
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base velocities to create elliptical path with coordinated lean.
        
        Ellipse navigation:
        - Narrow ends at phase ~0.125 and ~0.625 (high curvature)
        - Wide sides at phase ~0.375 and ~0.875 (low curvature)
        
        Yaw rate: peaks at narrow ends to create tight turns
        Roll rate: positive (right lean) at first narrow end, negative (left lean) at second
        """
        # Constant forward velocity
        vx = self.vx_forward
        
        # Yaw rate: sinusoidal with 2 peaks per cycle (narrow ends)
        # Peak at phase 0.125 and 0.625
        yaw_phase_shift = 0.25  # Shift to align peaks with narrow ends
        yaw_rate = self.yaw_rate_amplitude * np.sin(2 * np.pi * self.yaw_rate_freq * (phase + yaw_phase_shift))
        
        # Roll rate: sinusoidal with sign change to create alternating lean
        # Positive roll (right lean) in [0, 0.25], negative (left lean) in [0.5, 0.75]
        # Cosine gives: max at phase 0, min at phase 0.5
        roll_rate = self.roll_rate_amplitude * np.cos(2 * np.pi * self.roll_rate_freq * phase)
        
        # Set velocity commands in world frame
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

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        
        Trot gait pattern:
        - Stance phase (0 to duty): foot sweeps rearward
        - Swing phase (duty to 1): foot lifts, swings forward
        
        Lean compensation:
        - Adjust lateral foot position based on current roll angle
        - Adjust vertical position to maintain ground contact under roll
        """
        # Get leg-specific phase with diagonal trot offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start with base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Get current roll angle for lean compensation
        roll, pitch, yaw = quat_to_euler(self.root_quat)
        
        # Determine if this is a left or right leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Lateral compensation: shift foot outward under lean
        lateral_shift = 0.0
        if is_left_leg:
            lateral_shift = -self.lean_compensation * roll  # Negative y is left
            foot[1] += lateral_shift
        elif is_right_leg:
            lateral_shift = self.lean_compensation * roll  # Positive y is right
            foot[1] += lateral_shift
        
        # Vertical compensation: adjust height to maintain ground contact geometry
        # When body rolls, feet on the "down" side need vertical raise
        # Vertical offset proportional to lateral shift magnitude and roll angle
        if abs(roll) > 0.01:  # Apply compensation when roll is significant
            # For right legs during right lean (positive roll): need to raise foot
            # For left legs during left lean (negative roll): need to raise foot
            if is_right_leg and roll > 0:
                # Right leg, leaning right (down side) - raise foot
                foot[2] += abs(lateral_shift) * abs(np.tan(roll)) * 0.5
            elif is_left_leg and roll < 0:
                # Left leg, leaning left (down side) - raise foot
                foot[2] += abs(lateral_shift) * abs(np.tan(roll)) * 0.5
        
        if leg_phase < self.duty:
            # Stance phase: foot sweeps rearward relative to body
            progress = leg_phase / self.duty  # 0 to 1 through stance
            # Foot starts forward, sweeps back
            foot[0] += self.step_length * (0.5 - progress)
            
            # During stance, maintain ground contact by compensating for accumulated roll
            # Apply additional vertical adjustment based on current roll magnitude
            if abs(roll) > 0.01:
                stance_roll_compensation = 0.01 * abs(roll)  # Small additional lift during stance
                if (is_right_leg and roll > 0) or (is_left_leg and roll < 0):
                    foot[2] += stance_roll_compensation
        else:
            # Swing phase: foot lifts and swings forward
            progress = (leg_phase - self.duty) / (1.0 - self.duty)  # 0 to 1 through swing
            # Swing trajectory: parabolic height, linear forward
            swing_angle = np.pi * progress
            foot[0] += self.step_length * (progress - 0.5)
            foot[2] += self.step_height * np.sin(swing_angle)
        
        return foot