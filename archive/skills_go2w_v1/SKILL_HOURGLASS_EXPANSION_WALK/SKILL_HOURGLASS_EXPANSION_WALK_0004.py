from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_HOURGLASS_EXPANSION_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Hourglass Expansion Walk: A forward walking gait with rhythmic stance width oscillation.
    
    - All four legs converge toward and expand away from body centerline in synchronized pattern
    - Base height inversely correlates with stance width (rises during convergence, lowers during expansion)
    - Forward velocity maintained throughout with slight modulation
    - All feet maintain continuous ground contact (quasi-static gait)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.4
        
        # Base foot positions (neutral stance) - ensure above ground
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            foot_pos = v.copy()
            # Ensure minimum ground clearance in initial stance
            if foot_pos[2] < 0.02:
                foot_pos[2] = 0.02
            self.base_feet_pos_body[k] = foot_pos
        
        # Lateral width modulation parameters
        self.y_convergence_factor = 0.6  # Legs move inward to 60% (less extreme)
        self.y_expansion_factor = 1.4    # Legs move outward to 140% (less extreme)
        
        # Base height modulation (inverse to width)
        self.z_base_amplitude = 0.04     # Reduced vertical oscillation amplitude
        
        # Forward velocity modulation
        self.vx_baseline = 0.5
        self.vx_modulation = 0.15
        
        # Forward step management
        self.x_step_length = 0.12        # Reduced step length for smoother motion
        
        # Base state tracking
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.base_height_offset = 0.0    # Track integrated base height in world frame

    def update_base_motion(self, phase, dt):
        """
        Update base motion with forward velocity modulation and inverse height-width correlation.
        """
        
        # Forward velocity modulation: peaks during narrow stance
        phase_vx = (phase + 0.2) % 1.0
        vx = self.vx_baseline + self.vx_modulation * np.cos(2 * np.pi * phase_vx)
        
        # Vertical velocity: inversely correlated with stance width
        # Reduced magnitude during reconvergence phase to prevent contact loss
        if phase < 0.4:
            # Convergence phase: rising
            vz = self.z_base_amplitude * np.sin(np.pi * phase / 0.4)
        elif phase < 0.75:
            # Expansion phase: lowering (extended range)
            phase_exp = (phase - 0.4) / 0.35
            vz = -self.z_base_amplitude * np.sin(np.pi * phase_exp)
        else:
            # Reconvergence phase: rising (reduced magnitude, extended range)
            phase_reconv = (phase - 0.75) / 0.25
            vz = 0.7 * self.z_base_amplitude * np.sin(np.pi * phase_reconv)
        
        # No lateral or angular velocity
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        # Track cumulative base height offset
        self.base_height_offset += vz * dt
        
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
        Compute foot position in body frame with synchronized lateral width oscillation.
        
        Desynchronized forward stepping to maintain continuous contact:
        - FL, FR step during phase 0.75-0.9
        - RL, RR step during phase 0.9-1.0
        """
        
        base_foot = self.base_feet_pos_body[leg_name].copy()
        foot = base_foot.copy()
        
        # Determine leg characteristics
        is_left_leg = 'L' in leg_name and 'R' not in leg_name
        is_front_leg = leg_name.startswith('F')
        base_y_offset = abs(base_foot[1])
        
        # Lateral width oscillation (synchronized for all legs)
        width_phase = np.cos(2 * np.pi * phase)
        width_factor = self.y_convergence_factor + \
                       (self.y_expansion_factor - self.y_convergence_factor) * \
                       (1 - width_phase) / 2
        
        # Apply width modulation with correct sign
        if is_left_leg:
            foot[1] = base_y_offset * width_factor
        else:
            foot[1] = -base_y_offset * width_factor
        
        # Forward-rearward motion with desynchronized stepping
        if is_front_leg:
            # Front legs step during phase 0.75-0.9
            if phase < 0.75:
                # Gradual rearward translation
                foot[0] = base_foot[0] - self.x_step_length * (phase / 0.75)
            elif phase < 0.9:
                # Smooth forward step using raised cosine
                step_progress = (phase - 0.75) / 0.15
                step_blend = 0.5 * (1 - np.cos(np.pi * step_progress))
                foot[0] = base_foot[0] - self.x_step_length * (1 - step_blend)
            else:
                # Step complete, slight rearward drift until cycle restart
                foot[0] = base_foot[0] - self.x_step_length * (phase - 0.9) / 0.1
        else:
            # Rear legs step during phase 0.9-1.0
            if phase < 0.9:
                # Gradual rearward translation
                foot[0] = base_foot[0] - self.x_step_length * (phase / 0.9)
            else:
                # Smooth forward step using raised cosine
                step_progress = (phase - 0.9) / 0.1
                step_blend = 0.5 * (1 - np.cos(np.pi * step_progress))
                foot[0] = base_foot[0] - self.x_step_length * (1 - step_blend)
        
        # Vertical position adjustment: maintain ground contact as base height varies
        # As base rises in world frame, foot must move down in body frame to maintain ground contact
        # Correct sign: positive base_height_offset (base rose) requires negative adjustment (foot down in body)
        foot[2] = base_foot[2] - self.base_height_offset
        
        # Explicit ground contact safety constraint
        # Transform to world frame to check ground clearance
        foot_world = transform_body_to_world(foot, self.root_pos, self.root_quat)
        if foot_world[2] < 0.0:
            # Adjust body-frame z upward to enforce ground contact
            correction = -foot_world[2] + 0.001  # Add 1mm safety margin
            foot[2] += correction
        
        return foot