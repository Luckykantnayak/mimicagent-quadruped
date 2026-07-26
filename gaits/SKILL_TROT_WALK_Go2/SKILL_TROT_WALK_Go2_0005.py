from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_TROT_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal trot gait with continuous forward locomotion.
    
    Phase structure:
    - [0.0, 0.5]: FL+RR swing, FR+RL stance
    - [0.5, 1.0]: FR+RL swing, FL+RR stance
    
    Base motion:
    - Constant forward velocity (vx)
    - No lateral, vertical, or angular motion
    
    Leg motion:
    - Swing: smooth arc trajectory with step_length and step_height
    - Stance: foot moves rearward in body frame as base advances
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Gait parameters
        self.step_length = 0.12
        self.step_height = 0.06
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Diagonal pair phase offsets
        # Group 1 (FL, RR): swing in [0.0, 0.5], stance in [0.5, 1.0]
        # Group 2 (FR, RL): stance in [0.0, 0.5], swing in [0.5, 1.0]
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.phase_offsets[leg] = 0.5
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Forward velocity command
        self.vx_forward = 0.3

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity.
        No vertical or angular motion.
        """
        self.vel_world = np.array([self.vx_forward, 0.0, 0.0])
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
        Compute foot position in body frame based on phase.
        
        Each leg has a local phase determined by phase_offsets:
        - Swing phase: [0.0, 0.5) - foot follows arc trajectory forward
        - Stance phase: [0.5, 1.0) - foot moves rearward as base advances
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_phase < 0.5:
            # Swing phase: foot lifts, moves forward, and descends
            swing_progress = leg_phase / 0.5
            
            # Forward progression: from rear (-step_length/2) to front (+step_length/2)
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # Vertical arc: smooth parabolic trajectory
            # Peak at swing_progress = 0.5
            foot[2] += self.step_height * (1.0 - (2.0 * swing_progress - 1.0) ** 2)
            
        else:
            # Stance phase: foot in contact, moves rearward in body frame
            stance_progress = (leg_phase - 0.5) / 0.5
            
            # Rearward progression: from front (+step_length/2) to rear (-step_length/2)
            foot[0] += self.step_length * (0.5 - stance_progress)
        
        return foot