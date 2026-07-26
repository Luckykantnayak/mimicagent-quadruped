from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SHIMMY_SHAKE_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Shimmy-shake forward locomotion skill.
    
    - Base executes rapid lateral oscillations (shimmy) with forward bias
    - Legs make small incremental forward adjustments synchronized with weight shifts
    - All feet maintain contact throughout the cycle
    - High-frequency lateral oscillations enable dynamic stability and forward progression
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0  # 1 Hz cycle frequency
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time tracking
        self.t = 0.0
        
        # Base state (WORLD frame)
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Shimmy parameters
        self.lateral_vel_amp = 0.8  # High lateral velocity for sharp oscillations
        self.forward_vel_base = 0.12  # Small forward bias in early phases
        self.forward_vel_increased = 0.4  # Increased forward velocity in stepping phase
        
        # Leg stepping parameters
        self.micro_step_length = 0.02  # Small 2cm forward steps
        self.step_height = 0.0  # Feet stay on ground (stance throughout)
        
        # Phase-specific tracking for leg micro-steps
        self.leg_forward_offset = {leg: 0.0 for leg in leg_names}

    def update_base_motion(self, phase, dt):
        """
        Update base motion with rapid lateral oscillations and forward bias.
        
        Phase structure:
        - [0.0-0.1]: Shimmy right
        - [0.1-0.2]: Shimmy left
        - [0.2-0.3]: Shimmy right
        - [0.3-0.4]: Shimmy left
        - [0.4-0.7]: Continue oscillations with increased forward velocity
        - [0.7-1.0]: Dampen oscillations, maintain forward velocity
        """
        
        vx = 0.0
        vy = 0.0
        
        # Phase 0.0-0.1: Shimmy right
        if phase < 0.1:
            local_phase = phase / 0.1
            vx = self.forward_vel_base
            vy = self.lateral_vel_amp * np.sin(np.pi * local_phase)
            
        # Phase 0.1-0.2: Shimmy left
        elif phase < 0.2:
            local_phase = (phase - 0.1) / 0.1
            vx = self.forward_vel_base
            vy = -self.lateral_vel_amp * np.sin(np.pi * local_phase)
            
        # Phase 0.2-0.3: Shimmy right
        elif phase < 0.3:
            local_phase = (phase - 0.2) / 0.1
            vx = self.forward_vel_base
            vy = self.lateral_vel_amp * np.sin(np.pi * local_phase)
            
        # Phase 0.3-0.4: Shimmy left
        elif phase < 0.4:
            local_phase = (phase - 0.3) / 0.1
            vx = self.forward_vel_base
            vy = -self.lateral_vel_amp * np.sin(np.pi * local_phase)
            
        # Phase 0.4-0.7: Oscillate and step (higher frequency oscillations with increased forward velocity)
        elif phase < 0.7:
            local_phase = (phase - 0.4) / 0.3
            vx = self.forward_vel_increased
            # Continue rapid oscillations: ~3 cycles within this phase
            oscillation_phase = local_phase * 3.0
            vy = self.lateral_vel_amp * 0.8 * np.sin(2 * np.pi * oscillation_phase)
            
        # Phase 0.7-1.0: Dampen and advance
        else:
            local_phase = (phase - 0.7) / 0.3
            vx = self.forward_vel_increased
            # Exponential damping of lateral oscillations
            damping_factor = np.exp(-4.0 * local_phase)
            oscillation_phase = (phase - 0.7) / 0.3
            vy = self.lateral_vel_amp * 0.5 * damping_factor * np.sin(2 * np.pi * oscillation_phase)
        
        # Set velocity commands (WORLD frame)
        self.vel_world = np.array([vx, vy, 0.0])
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
        Compute foot position in BODY frame with small forward steps synchronized to weight shifts.
        
        Left-side legs (FL, RL) step forward during leftward weight shifts (phases 0.1-0.2, 0.3-0.4)
        Right-side legs (FR, RR) step forward during rightward weight shifts (phases 0.0-0.1, 0.2-0.3)
        All legs make multiple micro-steps during phase 0.4-0.7
        All legs maintain ground contact (stance throughout)
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        forward_offset = 0.0
        
        # Phase 0.0-0.1: Right legs step forward
        if phase < 0.1:
            if is_right_leg:
                local_phase = phase / 0.1
                forward_offset = self.micro_step_length * local_phase
                
        # Phase 0.1-0.2: Left legs step forward
        elif phase < 0.2:
            if is_right_leg:
                forward_offset = self.micro_step_length
            if is_left_leg:
                local_phase = (phase - 0.1) / 0.1
                forward_offset = self.micro_step_length * local_phase
                
        # Phase 0.2-0.3: Right legs step forward again
        elif phase < 0.3:
            if is_left_leg:
                forward_offset = self.micro_step_length
            if is_right_leg:
                local_phase = (phase - 0.2) / 0.1
                forward_offset = self.micro_step_length + self.micro_step_length * local_phase
                
        # Phase 0.3-0.4: Left legs step forward again
        elif phase < 0.4:
            if is_right_leg:
                forward_offset = 2.0 * self.micro_step_length
            if is_left_leg:
                local_phase = (phase - 0.3) / 0.1
                forward_offset = self.micro_step_length + self.micro_step_length * local_phase
                
        # Phase 0.4-0.7: Multiple small steps for all legs, alternating by side
        elif phase < 0.7:
            local_phase = (phase - 0.4) / 0.3
            
            # Base offset from previous phases
            base_offset = 2.0 * self.micro_step_length
            
            # 3 micro-steps within this phase, alternating left/right
            num_steps = 3
            step_phase = local_phase * num_steps
            step_index = int(step_phase)
            step_progress = step_phase - step_index
            
            if is_left_leg:
                # Left legs step on odd indices (0, 2)
                if step_index == 0:
                    forward_offset = base_offset + self.micro_step_length * step_progress
                elif step_index == 1:
                    forward_offset = base_offset + self.micro_step_length
                elif step_index == 2:
                    forward_offset = base_offset + self.micro_step_length + self.micro_step_length * step_progress
                else:
                    forward_offset = base_offset + 2.0 * self.micro_step_length
            else:  # Right legs
                # Right legs step on even/mixed indices
                if step_index == 0:
                    forward_offset = base_offset
                elif step_index == 1:
                    forward_offset = base_offset + self.micro_step_length * step_progress
                elif step_index == 2:
                    forward_offset = base_offset + self.micro_step_length
                else:
                    forward_offset = base_offset + self.micro_step_length + self.micro_step_length * step_progress
                    
        # Phase 0.7-1.0: Stabilize with minor forward adjustment
        else:
            local_phase = (phase - 0.7) / 0.3
            base_offset = 4.0 * self.micro_step_length  # Accumulated from previous phases
            forward_offset = base_offset + self.micro_step_length * local_phase
        
        # Apply forward offset in body frame (x-direction)
        foot[0] += forward_offset
        
        # Feet remain on ground (z unchanged)
        # Lateral position in body frame unchanged (y-axis stability)
        
        return foot