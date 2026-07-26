from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_TROT_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Trot gait with diagonal leg coordination for stable forward locomotion.
    
    - Diagonal pairs alternate: FL+RR in phase [0.0, 0.5], FR+RL in phase [0.5, 1.0]
    - Constant forward velocity throughout cycle
    - Swing phase uses smooth arc with moderate clearance
    - Stance phase tracks rearward in body frame as base advances
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0  # Hz, one full trot cycle per second
        
        # Trot gait parameters
        self.step_length = 0.12  # meters, forward reach in body frame during swing
        self.step_height = 0.06  # meters, swing clearance above ground
        
        # Base foot positions in body frame (neutral stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for diagonal trot coordination
        # FL and RR: stance in [0.0, 0.5], swing in [0.5, 1.0]
        # FR and RL: swing in [0.0, 0.5], stance in [0.5, 1.0]
        self.phase_offsets = {
            leg_names[0]: 0.0,  # FL
            leg_names[1]: 0.5,  # FR
            leg_names[2]: 0.5,  # RL
            leg_names[3]: 0.0,  # RR
        }
        
        # Base state (world frame)
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Constant velocity commands for trot
        self.vx_forward = 0.5  # m/s, moderate constant forward speed
        
    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward velocity and zero angular rates.
        Maintains level, straight motion throughout the trot cycle.
        """
        # Constant forward velocity in world frame
        self.vel_world = np.array([self.vx_forward, 0.0, 0.0])
        
        # Zero angular rates to keep base level and straight
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        # Integrate pose in world frame
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot trajectory in body frame for given leg and phase.
        
        Stance phase (duration 0.5): foot tracks rearward as body advances
        Swing phase (duration 0.5): foot lifts, swings forward in smooth arc, and places ahead
        """
        # Apply leg-specific phase offset for diagonal coordination
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from neutral foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_phase < 0.5:
            # Stance phase: foot moves rearward in body frame
            # progress goes from 0 (foot forward) to 1 (foot rearward)
            progress = leg_phase / 0.5
            
            # Foot starts at +step_length/2 ahead, ends at -step_length/2 behind
            foot[0] += self.step_length * (0.5 - progress)
            
            # Maintain ground contact (z = 0 offset from nominal)
            
        else:
            # Swing phase: foot lifts, swings forward, and places down
            # progress goes from 0 (liftoff) to 1 (touchdown)
            progress = (leg_phase - 0.5) / 0.5
            
            # Horizontal motion: foot moves from rear (-step_length/2) to front (+step_length/2)
            foot[0] += self.step_length * (progress - 0.5)
            
            # Vertical motion: smooth arc using sine for natural trajectory
            # Peak clearance at mid-swing (progress = 0.5)
            swing_angle = np.pi * progress
            foot[2] += self.step_height * np.sin(swing_angle)
        
        return foot