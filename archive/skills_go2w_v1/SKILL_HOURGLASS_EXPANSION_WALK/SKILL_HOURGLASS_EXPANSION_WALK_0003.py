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
        self.freq = 0.4  # Slower cycle for smooth width oscillation
        
        # Base foot positions (neutral stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Lateral width modulation parameters
        self.y_convergence_factor = 0.5  # Legs move inward to 50% of base lateral offset
        self.y_expansion_factor = 1.5    # Legs move outward to 150% of base lateral offset
        
        # Base height modulation (inverse to width)
        self.z_base_amplitude = 0.06     # Vertical oscillation amplitude
        
        # Forward velocity modulation
        self.vx_baseline = 0.5           # Steady forward velocity
        self.vx_modulation = 0.2         # Velocity variation amplitude
        
        # Forward step management to prevent rearward drift
        self.x_step_length = 0.15        # Forward repositioning distance
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base motion with forward velocity modulation and inverse height-width correlation.
        
        Phase-based velocity profile:
        - 0.0-0.2: convergence, vx increasing, vz upward
        - 0.2-0.4: narrow stance, vx peak, vz stabilizes/descends
        - 0.4-0.6: expansion, vx moderate, vz downward
        - 0.6-0.8: wide stance, vx steady, vz stabilizes
        - 0.8-1.0: reconvergence, vx increasing, vz upward
        """
        
        # Forward velocity modulation: peaks during narrow stance (phase ~0.3), 
        # moderates during wide stance (phase ~0.7)
        phase_vx = (phase + 0.2) % 1.0  # Shift phase for vx peak at narrow stance
        vx = self.vx_baseline + self.vx_modulation * np.cos(2 * np.pi * phase_vx)
        
        # Vertical velocity: inversely correlated with stance width
        # Width narrows (phase 0.0-0.4 and 0.8-1.0): vz positive (rising)
        # Width expands (phase 0.4-0.8): vz negative (lowering)
        if phase < 0.4:
            # Convergence phase: rising
            vz = self.z_base_amplitude * np.sin(np.pi * phase / 0.4)
        elif phase < 0.8:
            # Expansion phase: lowering
            phase_exp = (phase - 0.4) / 0.4
            vz = -self.z_base_amplitude * np.sin(np.pi * phase_exp)
        else:
            # Reconvergence phase: rising
            phase_reconv = (phase - 0.8) / 0.2
            vz = self.z_base_amplitude * np.sin(np.pi * phase_reconv)
        
        # No lateral or angular velocity
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
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
        
        All legs oscillate laterally in sync:
        - Phase 0.0-0.4: convergence (narrow stance)
        - Phase 0.4-0.8: expansion (wide stance)
        - Phase 0.8-1.0: reconvergence preparation
        
        Forward stepping occurs during reconvergence (phase 0.8-1.0) to prevent rearward drift.
        """
        
        base_foot = self.base_feet_pos_body[leg_name].copy()
        foot = base_foot.copy()
        
        # Determine lateral direction (left legs have positive y, right legs negative y)
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        base_y_offset = abs(base_foot[1])
        
        # Lateral width oscillation (synchronized for all legs)
        # Phase 0.0-0.5: narrow (convergence)
        # Phase 0.5-1.0: wide (expansion and reconvergence)
        width_phase = np.cos(2 * np.pi * phase)  # +1 at phase 0, -1 at phase 0.5, +1 at phase 1
        
        # Map width_phase [-1, 1] to [convergence_factor, expansion_factor]
        width_factor = self.y_convergence_factor + \
                       (self.y_expansion_factor - self.y_convergence_factor) * \
                       (1 - width_phase) / 2
        
        # Apply width modulation
        if is_left_leg:
            foot[1] = base_y_offset * width_factor
        else:
            foot[1] = -base_y_offset * width_factor
        
        # Forward-rearward motion in body frame
        # Legs translate rearward in body frame due to forward base motion
        # During phase 0.8-1.0, perform forward step to reset position
        if phase < 0.8:
            # Gradual rearward translation proportional to phase
            foot[0] = base_foot[0] - self.x_step_length * (phase / 0.8)
        else:
            # Forward step during reconvergence (phase 0.8-1.0)
            step_progress = (phase - 0.8) / 0.2
            foot[0] = base_foot[0] - self.x_step_length + self.x_step_length * step_progress
        
        # Vertical position adjustment: maintain ground contact as base height varies
        # Base rises during narrow stance (high width_phase), lowers during wide stance
        # Compensate by adjusting foot z in body frame inversely
        z_compensation = -self.z_base_amplitude * width_phase * 0.5
        foot[2] = base_foot[2] + z_compensation
        
        return foot