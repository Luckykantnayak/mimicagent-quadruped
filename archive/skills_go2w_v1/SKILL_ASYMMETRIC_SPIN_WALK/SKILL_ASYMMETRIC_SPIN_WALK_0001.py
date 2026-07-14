from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ASYMMETRIC_SPIN_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Asymmetric spin-walk gait with forward progression and leftward yaw drift.
    
    Phase structure:
    - [0.0, 0.4]: Left legs sweep wide/slow, right legs rotate tight/fast → leftward yaw
    - [0.4, 0.6]: Transition phase, rates and geometries converge
    - [0.6, 1.0]: Role reversal, right legs wide/slow, left legs tight/fast → brief rightward yaw
    
    Net effect over full cycle: sustained leftward yaw drift with forward locomotion.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Diagonal pair trot coordination
        # FL + RR swing together (group 1)
        # FR + RL swing together (group 2)
        self.phase_offsets = {
            leg_names[0]: 0.0,  # FL
            leg_names[1]: 0.5,  # FR
            leg_names[2]: 0.5,  # RL
            leg_names[3]: 0.0,  # RR
        }
        
        # Swing parameters
        self.duty_cycle = 0.5  # 50% stance, 50% swing for trot
        
        # Step geometry base parameters
        self.step_length_base = 0.12
        self.step_height_base = 0.08
        
        # Asymmetric sweep parameters
        self.wide_sweep_lateral = 0.06  # Wide arc lateral displacement
        self.tight_sweep_lateral = 0.02  # Tight rotation lateral displacement
        
        # Velocity scaling factors for rate asymmetry
        self.fast_rate_scale = 1.4  # Fast leg motion scaling
        self.slow_rate_scale = 0.7  # Slow leg motion scaling
        
        # Base motion parameters
        self.vx_forward = 0.8  # Sustained forward velocity
        self.base_height_oscillation = 0.03  # Vertical oscillation amplitude
        
        # Yaw rate parameters
        self.yaw_rate_left = -0.6  # Leftward yaw (negative)
        self.yaw_rate_right = 0.3  # Rightward yaw (positive, smaller magnitude)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        - Sustained forward velocity throughout
        - Phase-dependent yaw rate (leftward dominant, brief rightward reversal)
        - Vertical oscillation synchronized with diagonal pair alternation
        """
        # Forward velocity constant
        vx = self.vx_forward
        vy = 0.0
        
        # Vertical oscillation with diagonal pair alternation
        vz = self.base_height_oscillation * np.sin(2 * np.pi * phase)
        
        # Phase-dependent yaw rate
        if phase < 0.4:
            # Left legs wide/slow, right legs tight/fast → leftward yaw
            yaw_rate = self.yaw_rate_left
        elif phase < 0.6:
            # Transition: smooth interpolation from left to right yaw
            transition_progress = (phase - 0.4) / 0.2
            yaw_rate = self.yaw_rate_left * (1 - transition_progress) + self.yaw_rate_right * transition_progress
        else:
            # Role reversal: brief rightward yaw
            yaw_rate = self.yaw_rate_right
        
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame with asymmetric sweep geometry.
        
        Left legs (FL, RL): wide/slow in phase 0-0.4, tight/fast in phase 0.6-1.0
        Right legs (FR, RR): tight/fast in phase 0-0.4, wide/slow in phase 0.6-1.0
        """
        # Get leg-specific phase with diagonal pair offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Determine if left or right leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front_leg = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Phase-dependent asymmetry parameters
        if phase < 0.4:
            # Phase 0-0.4: left wide/slow, right tight/fast
            if is_left_leg:
                lateral_sweep = self.wide_sweep_lateral
                rate_scale = self.slow_rate_scale
            else:
                lateral_sweep = self.tight_sweep_lateral
                rate_scale = self.fast_rate_scale
        elif phase < 0.6:
            # Phase 0.4-0.6: transition, parameters converge
            transition_progress = (phase - 0.4) / 0.2
            if is_left_leg:
                lateral_sweep = self.wide_sweep_lateral * (1 - transition_progress) + self.tight_sweep_lateral * transition_progress
                rate_scale = self.slow_rate_scale * (1 - transition_progress) + self.fast_rate_scale * transition_progress
            else:
                lateral_sweep = self.tight_sweep_lateral * (1 - transition_progress) + self.wide_sweep_lateral * transition_progress
                rate_scale = self.fast_rate_scale * (1 - transition_progress) + self.slow_rate_scale * transition_progress
        else:
            # Phase 0.6-1.0: role reversal, left tight/fast, right wide/slow
            if is_left_leg:
                lateral_sweep = self.tight_sweep_lateral
                rate_scale = self.fast_rate_scale
            else:
                lateral_sweep = self.wide_sweep_lateral
                rate_scale = self.slow_rate_scale
        
        # Swing/stance determination
        if leg_phase < self.duty_cycle:
            # Stance phase: foot moves backward relative to body
            stance_progress = leg_phase / self.duty_cycle
            foot[0] += self.step_length_base * (0.5 - stance_progress) * rate_scale
        else:
            # Swing phase: foot lifts, swings forward with lateral arc
            swing_progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)
            
            # Forward displacement
            foot[0] += self.step_length_base * (swing_progress - 0.5) * rate_scale
            
            # Lateral sweep (asymmetric)
            lateral_sign = 1.0 if is_left_leg else -1.0
            foot[1] += lateral_sign * lateral_sweep * np.sin(np.pi * swing_progress)
            
            # Vertical lift
            foot[2] += self.step_height_base * np.sin(np.pi * swing_progress)
        
        return foot