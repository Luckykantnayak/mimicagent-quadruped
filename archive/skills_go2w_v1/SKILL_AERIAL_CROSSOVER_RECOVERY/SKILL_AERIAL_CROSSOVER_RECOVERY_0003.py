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
        self.skating_step_height = 0.05
        
        # Aerial crossover parameters
        self.aerial_lift_height = 0.15
        self.crossover_lateral_amplitude = 0.20
        
        # Base velocity parameters
        self.vx_entry = 0.8
        self.vx_aerial = 0.9
        self.vx_resume = 1.2
        self.vz_aerial_up = 0.25
        self.vz_aerial_down = -0.25
        
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
        
        # Track integrated base z-displacement for compensation
        self.base_z_offset = 0.0
        self.prev_phase = 0.0

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
            vz = 0.0
            
        # Phase [0.2, 0.4]: first_diagonal_aerial (FL+RR airborne)
        elif phase < 0.4:
            vx = self.vx_aerial
            local_phase = (phase - 0.2) / 0.2
            vy = 0.05 * np.sin(np.pi * local_phase)
            # Smoother vertical profile
            if local_phase < 0.4:
                vz = self.vz_aerial_up * np.sin(np.pi * local_phase / 0.4)
            elif local_phase < 0.6:
                vz = 0.0
            else:
                vz = self.vz_aerial_down * np.sin(np.pi * (local_phase - 0.6) / 0.4)
            roll_rate = self.roll_rate_aerial * (1.0 - abs(2.0 * local_phase - 1.0))
            yaw_rate = self.yaw_rate_aerial * (1.0 - abs(2.0 * local_phase - 1.0))
            
        # Phase [0.4, 0.45]: transition_landing
        elif phase < 0.45:
            vx = self.vx_aerial
            local_phase = (phase - 0.4) / 0.05
            vz = -0.12 * (1.0 - local_phase)
            
        # Phase [0.45, 0.65]: second_diagonal_aerial (FR+RL airborne)
        elif phase < 0.65:
            vx = self.vx_aerial
            local_phase = (phase - 0.45) / 0.2
            vy = -0.05 * np.sin(np.pi * local_phase)
            # Smoother vertical profile
            if local_phase < 0.4:
                vz = self.vz_aerial_up * np.sin(np.pi * local_phase / 0.4)
            elif local_phase < 0.6:
                vz = 0.0
            else:
                vz = self.vz_aerial_down * np.sin(np.pi * (local_phase - 0.6) / 0.4)
            roll_rate = -self.roll_rate_aerial * (1.0 - abs(2.0 * local_phase - 1.0))
            yaw_rate = self.yaw_rate_aerial * 0.8 * (1.0 - abs(2.0 * local_phase - 1.0))
            
        # Phase [0.65, 0.75]: recovery_landing
        elif phase < 0.75:
            vx = self.vx_aerial * 1.1
            local_phase = (phase - 0.65) / 0.1
            vz = -0.12 * (1.0 - local_phase)
            yaw_rate = 0.1 * (1.0 - local_phase)
            
        # Phase [0.75, 1.0]: skating_resume
        else:
            vx = self.vx_resume
            vz = 0.0
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        # Track integrated vertical displacement
        self.base_z_offset += vz * dt
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        self.prev_phase = phase

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        Synchronized with base motion to prevent ground penetration.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine which group the leg belongs to
        is_group_1 = leg_name.startswith('FL') or leg_name.startswith('RR')
        is_group_2 = leg_name.startswith('FR') or leg_name.startswith('RL')
        
        # Lateral sign for crossover motions
        lateral_sign = 1.0 if leg_name.startswith('FL') or leg_name.startswith('FR') else -1.0
        
        # Compute forward displacement to match base velocity integration
        # Stance feet must sweep backward to stay stationary in world frame
        
        # Phase [0.0, 0.2]: skating_entry - all feet in stance
        if phase < 0.2:
            local_phase = phase / 0.2
            # Match integrated base displacement: vx * duration = 0.8 * 0.2 = 0.16m
            phase_displacement = self.vx_entry * 0.2
            foot[0] -= phase_displacement * (local_phase - 0.5)
            foot[2] += 0.005 * np.sin(2.0 * np.pi * local_phase)
            
        # Phase [0.2, 0.4]: first_diagonal_aerial
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            phase_displacement = self.vx_aerial * 0.2
            
            if is_group_1:  # FL+RR airborne, execute crossover
                # Lift upward with smooth profile
                lift_profile = np.sin(np.pi * local_phase)
                foot[2] += self.aerial_lift_height * lift_profile
                # Compensate for base vertical motion
                foot[2] -= self.base_z_offset
                # Crossover wave: arc inward across midline then back out
                crossover_offset = self.crossover_lateral_amplitude * np.sin(2 * np.pi * local_phase)
                foot[1] += lateral_sign * crossover_offset
                # Forward sweep during aerial with smooth acceleration
                foot[0] += phase_displacement * 0.5 * (local_phase - 0.5)
            else:  # FR+RL maintain stance
                # Sweep backward to match base forward motion
                foot[0] -= phase_displacement * local_phase
                # Ensure ground contact - small compliance
                foot[2] += 0.002
                
        # Phase [0.4, 0.45]: transition_landing
        elif phase < 0.45:
            local_phase = (phase - 0.4) / 0.05
            phase_displacement = self.vx_aerial * 0.05
            # All feet grounded, smooth transition
            foot[0] -= phase_displacement * local_phase
            foot[2] -= self.base_z_offset
            foot[2] += 0.002
            
        # Phase [0.45, 0.65]: second_diagonal_aerial
        elif phase < 0.65:
            local_phase = (phase - 0.45) / 0.2
            phase_displacement = self.vx_aerial * 0.2
            
            if is_group_2:  # FR+RL airborne, execute crossover
                # Lift upward with smooth profile
                lift_profile = np.sin(np.pi * local_phase)
                foot[2] += self.aerial_lift_height * lift_profile
                # Compensate for base vertical motion
                foot[2] -= self.base_z_offset
                # Crossover wave: arc inward across midline then back out
                crossover_offset = self.crossover_lateral_amplitude * np.sin(2 * np.pi * local_phase)
                foot[1] += lateral_sign * crossover_offset
                # Forward sweep during aerial with smooth acceleration
                foot[0] += phase_displacement * 0.5 * (local_phase - 0.5)
            else:  # FL+RR maintain stance
                # Sweep backward to match base forward motion
                foot[0] -= phase_displacement * local_phase
                # Ensure ground contact
                foot[2] += 0.002
                
        # Phase [0.65, 0.75]: recovery_landing
        elif phase < 0.75:
            local_phase = (phase - 0.65) / 0.1
            phase_displacement = (self.vx_aerial * 1.1) * 0.1
            # All feet grounded, re-synchronizing
            foot[0] -= phase_displacement * local_phase
            foot[2] -= self.base_z_offset * (1.0 - local_phase)
            foot[2] += 0.002
            
        # Phase [0.75, 1.0]: skating_resume
        else:
            local_phase = (phase - 0.75) / 0.25
            phase_displacement = self.vx_resume * 0.25
            
            # Resume skating wave with enhanced velocity
            # Alternating diagonal pattern with coordinated swing phases
            if is_group_1:
                foot[0] -= phase_displacement * (local_phase - 0.5)
                # Swing phase for smooth skating motion
                if local_phase > 0.5 and local_phase < 0.9:
                    swing_phase = (local_phase - 0.5) / 0.4
                    foot[2] += self.skating_step_height * np.sin(np.pi * swing_phase)
                else:
                    foot[2] += 0.002
            else:
                foot[0] -= phase_displacement * (local_phase - 0.3)
                # Offset swing phase for alternating pattern
                if local_phase > 0.1 and local_phase < 0.5:
                    swing_phase = (local_phase - 0.1) / 0.4
                    foot[2] += self.skating_step_height * np.sin(np.pi * swing_phase)
                else:
                    foot[2] += 0.002
        
        return foot