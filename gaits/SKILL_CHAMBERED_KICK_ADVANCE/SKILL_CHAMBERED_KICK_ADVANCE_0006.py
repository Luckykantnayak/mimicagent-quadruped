from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_CHAMBERED_KICK_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Chambered Kick Advance: Sequential leg chamber-extend kicks for forward locomotion.
    
    Motion sequence: RL → RR → FL → FR, each leg chambers (retracts with bent knee)
    then explosively extends forward-downward. Base surges forward during extensions.
    Three-point support maintained throughout cycle via sequential activation.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions (neutral stance in body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Chamber and extension parameters
        self.chamber_height = 0.12  # Vertical lift during chamber
        self.chamber_retract = 0.08  # Horizontal retraction toward body
        self.extension_forward = 0.15  # Forward reach during extension
        self.extension_down = -0.05  # Downward reach during extension
        
        # Base velocity parameters
        self.surge_velocity = 1.2  # High forward velocity during extensions
        self.drift_velocity = 0.3  # Low forward velocity during chambers
        self.moderate_velocity = 0.7  # Moderate velocity for front legs
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Phase boundaries for each leg's kick cycle
        self.leg_phase_map = {
            'RL': {'chamber_start': 0.0, 'chamber_end': 0.15, 'extend_end': 0.3, 'cycle_end': 1.0},
            'RR': {'chamber_start': 0.3, 'chamber_end': 0.45, 'extend_end': 0.6, 'cycle_end': 1.0},
            'FL': {'chamber_start': 0.6, 'chamber_end': 0.675, 'extend_end': 0.75, 'cycle_end': 1.0},
            'FR': {'chamber_start': 0.75, 'chamber_end': 0.825, 'extend_end': 0.9, 'cycle_end': 1.0},
        }

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent forward velocity surges.
        Surges occur during leg extensions; reduced velocity during chambers.
        """
        # Determine forward velocity based on phase
        if 0.15 <= phase < 0.3:  # RL extension surge
            vx = self.surge_velocity
        elif 0.45 <= phase < 0.6:  # RR extension surge
            vx = self.surge_velocity
        elif 0.6 <= phase < 0.75:  # FL chamber-extend
            vx = self.moderate_velocity
        elif 0.75 <= phase < 0.9:  # FR chamber-extend
            vx = self.moderate_velocity
        elif 0.9 <= phase < 1.0:  # Neutral reset (deceleration)
            progress = (phase - 0.9) / 0.1
            vx = self.moderate_velocity * (1.0 - progress)
        else:  # Chamber phases (low drift)
            vx = self.drift_velocity
        
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on leg's kick cycle phase.
        Each leg chambers (lifts and retracts), then extends forward-downward.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Identify leg group (rear vs front)
        is_rear = leg_name.startswith('R')
        
        # Get phase boundaries for this leg
        phase_map = self.leg_phase_map[leg_name]
        chamber_start = phase_map['chamber_start']
        chamber_end = phase_map['chamber_end']
        extend_end = phase_map['extend_end']
        
        # Determine leg state based on phase
        if chamber_start <= phase < chamber_end:
            # CHAMBER: Retract close to body with knee bent high
            progress = (phase - chamber_start) / (chamber_end - chamber_start)
            # Smooth interpolation using sinusoidal easing
            t = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            foot = base_pos.copy()
            foot[0] -= self.chamber_retract * t  # Retract rearward
            foot[2] += self.chamber_height * np.sin(np.pi * progress)  # Lift upward (arc)
            
        elif chamber_end <= phase < extend_end:
            # EXTEND: Explosive forward-downward extension
            progress = (phase - chamber_end) / (extend_end - chamber_end)
            # Smooth interpolation
            t = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            foot = base_pos.copy()
            # Transition from chambered position to extended position
            chamber_x = base_pos[0] - self.chamber_retract
            chamber_z = base_pos[2] + self.chamber_height * np.sin(np.pi * 1.0)  # Peak height
            
            extend_x = base_pos[0] + self.extension_forward
            extend_z = base_pos[2] + self.extension_down
            
            foot[0] = chamber_x + (extend_x - chamber_x) * t
            foot[2] = chamber_z + (extend_z - chamber_z) * t
            
        else:
            # STANCE: Return to neutral and maintain support
            # Smooth transition back to base position
            if phase < chamber_start:
                # Before kick cycle starts: neutral stance
                foot = base_pos.copy()
            else:
                # After extension: recover to neutral
                recovery_duration = 0.1  # 10% of cycle for recovery
                if phase < extend_end + recovery_duration:
                    progress = (phase - extend_end) / recovery_duration
                    t = 0.5 - 0.5 * np.cos(np.pi * progress)
                    
                    extend_x = base_pos[0] + self.extension_forward
                    extend_z = base_pos[2] + self.extension_down
                    
                    foot = base_pos.copy()
                    foot[0] = extend_x + (base_pos[0] - extend_x) * t
                    foot[2] = extend_z + (base_pos[2] - extend_z) * t
                else:
                    # Fully recovered to neutral stance
                    foot = base_pos.copy()
        
        return foot