from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_AERIAL_CROSSOVER_RECOVERY_MotionGenerator(BaseMotionGenerator):
    """
    Aerial Crossover Recovery Skill
    
    A forward skating motion interrupted by brief aerial phases where diagonal leg pairs 
    lift simultaneously, execute crossover wave gestures mid-air to reconfigure momentum 
    and orientation, then land back into phase to resume skating with enhanced velocity 
    and directional shift.
    
    Phase Structure:
    - [0.0, 0.2]: skating_entry - all feet grounded, building forward speed
    - [0.2, 0.4]: first_diagonal_aerial - FL+RR airborne, FR+RL stance
    - [0.4, 0.45]: transition_landing - all feet grounded
    - [0.45, 0.65]: second_diagonal_aerial - FR+RL airborne, FL+RR stance
    - [0.65, 0.75]: recovery_landing - all feet grounded
    - [0.75, 1.0]: skating_resume - enhanced skating with adjusted heading
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Skating parameters
        self.skating_step_length = 0.12
        self.skating_step_height = 0.05
        
        # Aerial crossover parameters
        self.aerial_lift_height = 0.15
        self.crossover_lateral_amplitude = 0.20
        
        # Base velocity parameters
        self.vx_entry = 0.8
        self.vx_aerial = 0.9
        self.vx_resume = 1.2
        self.vz_aerial_up = 0.3
        self.vz_aerial_down = -0.3
        
        # Angular velocity parameters
        self.roll_rate_aerial = 0.4
        self.yaw_rate_aerial = 0.6
        
        # Diagonal pair groupings
        self.group_1 = []  # FL, RR
        self.group_2 = []  # FR, RL
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.group_1.append(leg)
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.group_2.append(leg)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands according to phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase [0.0, 0.2]: skating_entry
        if phase < 0.2:
            vx = self.vx_entry
            vy = 0.0
            vz = 0.0
            roll_rate = 0.0
            pitch_rate = 0.0
            yaw_rate = 0.0
            
        # Phase [0.2, 0.4]: first_diagonal_aerial (FL+RR airborne)
        elif phase < 0.4:
            vx = self.vx_aerial
            # Slight lateral compensation
            local_phase = (phase - 0.2) / 0.2
            vy = 0.05 * np.sin(np.pi * local_phase)
            # Vertical lift at start, descent at end
            if local_phase < 0.5:
                vz = self.vz_aerial_up * (1.0 - 2.0 * local_phase)
            else:
                vz = self.vz_aerial_down * (2.0 * local_phase - 1.0)
            # Roll compensation for asymmetric support (FR+RL diagonal)
            roll_rate = self.roll_rate_aerial
            pitch_rate = 0.0
            # Initiate yaw shift
            yaw_rate = self.yaw_rate_aerial
            
        # Phase [0.4, 0.45]: transition_landing
        elif phase < 0.45:
            vx = self.vx_aerial
            vy = 0.0
            vz = -0.2  # Settle down
            roll_rate = 0.0
            pitch_rate = 0.0
            yaw_rate = 0.0
            
        # Phase [0.45, 0.65]: second_diagonal_aerial (FR+RL airborne)
        elif phase < 0.65:
            vx = self.vx_aerial
            # Opposite lateral compensation
            local_phase = (phase - 0.45) / 0.2
            vy = -0.05 * np.sin(np.pi * local_phase)
            # Vertical lift at start, descent at end
            if local_phase < 0.5:
                vz = self.vz_aerial_up * (1.0 - 2.0 * local_phase)
            else:
                vz = self.vz_aerial_down * (2.0 * local_phase - 1.0)
            # Roll compensation for asymmetric support (FL+RR diagonal)
            roll_rate = -self.roll_rate_aerial
            pitch_rate = 0.0
            # Continue or enhance yaw shift
            yaw_rate = self.yaw_rate_aerial * 0.8
            
        # Phase [0.65, 0.75]: recovery_landing
        elif phase < 0.75:
            vx = self.vx_aerial * 1.1
            vy = 0.0
            vz = -0.2  # Settle down
            roll_rate = 0.0
            pitch_rate = 0.0
            yaw_rate = 0.1  # Finalize heading adjustment
            
        # Phase [0.75, 1.0]: skating_resume
        else:
            vx = self.vx_resume
            vy = 0.0
            vz = 0.0
            roll_rate = 0.0
            pitch_rate = 0.0
            yaw_rate = 0.0
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
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
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine which group the leg belongs to
        is_group_1 = leg_name.startswith('FL') or leg_name.startswith('RR')
        is_group_2 = leg_name.startswith('FR') or leg_name.startswith('RL')
        
        # Phase [0.0, 0.2]: skating_entry - all feet in stance
        if phase < 0.2:
            local_phase = phase / 0.2
            foot[0] -= self.skating_step_length * (local_phase - 0.5)
            
        # Phase [0.2, 0.4]: first_diagonal_aerial
        elif phase < 0.4:
            if is_group_1:  # FL+RR airborne, execute crossover
                local_phase = (phase - 0.2) / 0.2
                # Lift upward
                foot[2] += self.aerial_lift_height * np.sin(np.pi * local_phase)
                # Crossover wave: arc inward across midline then back out
                lateral_sign = 1.0 if leg_name.startswith('FL') else -1.0
                crossover_offset = self.crossover_lateral_amplitude * np.sin(2 * np.pi * local_phase)
                foot[1] += lateral_sign * crossover_offset
                # Forward sweep during aerial
                foot[0] += self.skating_step_length * (local_phase - 0.5)
            else:  # FR+RL maintain stance
                local_phase = (phase - 0.2) / 0.2
                foot[0] -= self.skating_step_length * (local_phase + 0.3)
                
        # Phase [0.4, 0.45]: transition_landing
        elif phase < 0.45:
            local_phase = (phase - 0.4) / 0.05
            # All feet grounded, smooth transition
            foot[0] -= self.skating_step_length * 0.5
            
        # Phase [0.45, 0.65]: second_diagonal_aerial
        elif phase < 0.65:
            if is_group_2:  # FR+RL airborne, execute crossover
                local_phase = (phase - 0.45) / 0.2
                # Lift upward
                foot[2] += self.aerial_lift_height * np.sin(np.pi * local_phase)
                # Crossover wave: arc inward across midline then back out
                lateral_sign = 1.0 if leg_name.startswith('FR') else -1.0
                crossover_offset = self.crossover_lateral_amplitude * np.sin(2 * np.pi * local_phase)
                foot[1] += lateral_sign * crossover_offset
                # Forward sweep during aerial
                foot[0] += self.skating_step_length * (local_phase - 0.5)
            else:  # FL+RR maintain stance
                local_phase = (phase - 0.45) / 0.2
                foot[0] -= self.skating_step_length * (local_phase + 0.3)
                
        # Phase [0.65, 0.75]: recovery_landing
        elif phase < 0.75:
            local_phase = (phase - 0.65) / 0.1
            # All feet grounded, re-synchronizing
            foot[0] -= self.skating_step_length * (0.3 * (1.0 - local_phase))
            
        # Phase [0.75, 1.0]: skating_resume
        else:
            local_phase = (phase - 0.75) / 0.25
            # Resume skating wave with enhanced velocity
            # Alternating diagonal pattern
            if is_group_1:
                foot[0] -= self.skating_step_length * 1.2 * (local_phase - 0.5)
                # Slight swing if needed
                if local_phase > 0.6:
                    swing_phase = (local_phase - 0.6) / 0.4
                    foot[2] += self.skating_step_height * np.sin(np.pi * swing_phase)
            else:
                foot[0] -= self.skating_step_length * 1.2 * (local_phase - 0.3)
                if local_phase > 0.3 and local_phase < 0.7:
                    swing_phase = (local_phase - 0.3) / 0.4
                    foot[2] += self.skating_step_height * np.sin(np.pi * swing_phase)
        
        return foot