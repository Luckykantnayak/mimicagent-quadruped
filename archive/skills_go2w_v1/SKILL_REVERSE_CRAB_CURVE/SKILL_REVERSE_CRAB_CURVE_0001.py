from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_REVERSE_CRAB_CURVE_MotionGenerator(BaseMotionGenerator):
    """
    Reverse crab curve gait: Robot travels backward along curved path while
    body orientation continuously rotates, maintaining perpendicular relationship
    to travel direction. Diagonal leg coordination with outer legs taking larger
    strides to create curved trajectory.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.6
        
        # Gait timing parameters
        self.duty_cycle = 0.67
        self.swing_phase_duration = 0.17
        
        # Step parameters
        self.step_height = 0.07
        self.outer_leg_step_length = 0.14
        self.inner_leg_step_length = 0.08
        self.outer_leg_lateral_extension = 0.06
        self.inner_leg_lateral_extension = 0.02
        
        # Base foot positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for diagonal coordination
        # Group 1 (FL+RR): swings at 0.16-0.33 and 0.83-1.0
        # Group 2 (FR+RL): swings at 0.50-0.67
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.16
            else:
                self.phase_offsets[leg] = 0.50
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters
        self.vx_backward = -0.3
        self.vy_max = 0.15
        self.yaw_rate = -0.8

    def update_base_motion(self, phase, dt):
        """
        Update base using continuous backward velocity with transitioning lateral
        velocity and constant negative yaw rate to create curved backward motion
        with perpendicular body orientation.
        """
        vx = self.vx_backward
        
        # Lateral velocity transitions: left -> zero -> right across cycle
        # Phase 0.0-0.33: leftward
        # Phase 0.33-0.67: transition from left to right
        # Phase 0.67-1.0: rightward
        if phase < 0.33:
            vy = -self.vy_max
        elif phase < 0.67:
            transition_phase = (phase - 0.33) / 0.34
            vy = -self.vy_max + 2.0 * self.vy_max * transition_phase
        else:
            vy = self.vy_max
        
        # Continuous negative yaw rate for counterclockwise rotation
        yaw_rate = self.yaw_rate
        
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with diagonal coordination and
        asymmetric stride lengths for curved trajectory.
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is an outer or inner leg based on phase and leg position
        # FL and RR act as outer legs during leftward curve (phase 0.0-0.5)
        # FR and RL act as outer legs during rightward curve (phase 0.5-1.0)
        is_outer_leg = False
        if phase < 0.5:
            if leg_name.startswith('FL') or leg_name.startswith('RR'):
                is_outer_leg = True
        else:
            if leg_name.startswith('FR') or leg_name.startswith('RL'):
                is_outer_leg = True
        
        # Select step parameters based on outer/inner designation
        if is_outer_leg:
            step_length = self.outer_leg_step_length
            lateral_extension = self.outer_leg_lateral_extension
        else:
            step_length = self.inner_leg_step_length
            lateral_extension = self.inner_leg_lateral_extension
        
        # Determine swing timing for diagonal groups
        # Group 1 (FL, RR): swing at local phase 0.0-0.17 (wrapping from offset 0.16)
        # Group 2 (FR, RL): swing at local phase 0.0-0.17 (wrapping from offset 0.50)
        in_swing = leg_phase < self.swing_phase_duration
        
        if in_swing:
            # Swing phase: arc trajectory with forward-to-backward motion in body frame
            swing_progress = leg_phase / self.swing_phase_duration
            
            # Forward-to-backward motion (positive to negative x in body frame)
            foot[0] += step_length * (0.5 - swing_progress)
            
            # Lateral extension for outer legs
            if leg_name.startswith('FL') or leg_name.startswith('RL'):
                foot[1] += lateral_extension * np.sin(np.pi * swing_progress)
            else:
                foot[1] -= lateral_extension * np.sin(np.pi * swing_progress)
            
            # Vertical arc
            foot[2] += self.step_height * np.sin(np.pi * swing_progress)
        else:
            # Stance phase: foot drifts backward in body frame due to forward base motion
            stance_progress = (leg_phase - self.swing_phase_duration) / (1.0 - self.swing_phase_duration)
            
            # Backward drift in body frame (simulates ground contact with backward-moving body)
            foot[0] -= step_length * stance_progress
        
        return foot