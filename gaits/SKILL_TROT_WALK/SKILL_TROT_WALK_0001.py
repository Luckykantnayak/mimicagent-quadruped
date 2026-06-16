from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_TROT_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal trot gait with continuous forward locomotion.
    
    - FL and RR swing together (phase 0.0-0.5) while FR and RL provide stance support
    - FR and RL swing together (phase 0.5-1.0) while FL and RR provide stance support
    - Base maintains constant forward velocity throughout
    - Duty cycle is 0.5 for each leg
    - Foot trajectories expressed in BODY frame
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        self.duty = 0.5  # 50% duty cycle for trot gait
        self.step_length = 0.12  # Forward displacement during swing
        self.step_height = 0.06  # Peak height during swing arc
        
        # Base foot positions in BODY frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for diagonal trot coordination
        # FL and RR swing together (offset 0.0)
        # FR and RL swing together (offset 0.5)
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
        
        # Constant forward velocity command
        self.vx_command = 0.5  # Forward velocity in m/s
        
    def update_base_motion(self, phase, dt):
        """
        Update base pose with constant forward velocity.
        No angular velocity to maintain straight-line motion.
        """
        # Constant forward velocity throughout the gait cycle
        self.vel_world = np.array([self.vx_command, 0.0, 0.0])
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
        Compute foot position in BODY frame based on phase.
        
        Swing phase (duty=0.5):
          - Foot lifts from rear position, arcs forward with peak height at mid-swing
        Stance phase (duty=0.5):
          - Foot sweeps backward in body frame as base moves forward
        """
        # Apply phase offset for diagonal coordination
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start with base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_phase < self.duty:
            # Swing phase: foot arcs forward and upward
            swing_progress = leg_phase / self.duty
            
            # Horizontal motion: sweep from rear (-step_length/2) to forward (+step_length/2)
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # Vertical motion: sinusoidal arc with peak at mid-swing
            swing_angle = np.pi * swing_progress
            foot[2] += self.step_height * np.sin(swing_angle)
            
        else:
            # Stance phase: foot sweeps backward in body frame
            stance_progress = (leg_phase - self.duty) / (1.0 - self.duty)
            
            # Foot moves from forward position to rear position
            foot[0] += self.step_length * (0.5 - stance_progress)
        
        return foot