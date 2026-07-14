from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ESCALATOR_CLIMB_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Escalator climbing motion with continuous forward and upward base velocity.
    
    - Base moves with constant forward (vx) and upward (vz) velocity throughout cycle
    - Diagonal trot gait: FL+RR alternate with FR+RL
    - 50% duty cycle per leg group
    - Stance phase: foot sweeps backward-downward in body frame
    - Swing phase: foot arcs forward-upward in body frame
    - No flight phase: always two feet in contact
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Diagonal trot coordination: FL+RR in phase, FR+RL anti-phase
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0  # Group 1: stance 0.0-0.5, swing 0.5-1.0
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.phase_offsets[leg] = 0.5  # Group 2: swing 0.0-0.5, stance 0.5-1.0
        
        # Gait parameters
        self.duty_cycle = 0.5  # 50% stance, 50% swing per leg
        
        # Stance phase foot motion (backward-downward sweep in body frame)
        self.stance_sweep_x = 0.12  # Forward-backward sweep range during stance
        self.stance_sweep_z = 0.06  # Vertical sweep range during stance (downward motion)
        
        # Swing phase foot motion (forward-upward arc in body frame)
        self.swing_step_x = 0.12   # Forward reach during swing
        self.swing_height = 0.10   # Peak height during swing arc
        self.swing_step_z = 0.06   # Upward reach during swing to match climb
        
        # Base velocity parameters (world frame)
        self.vx_climb = 0.4   # Forward velocity (m/s)
        self.vz_climb = 0.25  # Upward velocity (m/s) - creates ~32 degree climb angle
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Initialize velocity commands
        self.vel_world = np.array([self.vx_climb, 0.0, self.vz_climb])
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Base climbs continuously forward and upward at constant velocity.
        No angular rates - body orientation remains stable.
        """
        # Constant forward and upward velocity throughout entire cycle
        self.vel_world = np.array([self.vx_climb, 0.0, self.vz_climb])
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
        Compute foot position in body frame for given leg and phase.
        
        Stance phase (leg_phase 0.0-0.5):
          - Foot sweeps backward (negative x) and downward (negative z) in body frame
          - This maintains ground contact as base climbs forward-upward
        
        Swing phase (leg_phase 0.5-1.0):
          - Foot lifts up and arcs forward-upward
          - Prepares for next stance by positioning ahead and higher
        """
        # Apply phase offset for diagonal coordination
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_phase < self.duty_cycle:
            # STANCE PHASE (0.0 - 0.5)
            # Foot is in contact, sweeps backward-downward as base climbs
            progress = leg_phase / self.duty_cycle  # 0.0 -> 1.0 during stance
            
            # Backward sweep: foot moves from front to back of stance
            # At progress=0: foot at forward position (+0.5 * sweep)
            # At progress=1: foot at rear position (-0.5 * sweep)
            foot[0] += self.stance_sweep_x * (0.5 - progress)
            
            # Downward sweep: foot moves down to maintain contact during climb
            # At progress=0: foot at higher position
            # At progress=1: foot at lower position
            foot[2] -= self.stance_sweep_z * progress
            
        else:
            # SWING PHASE (0.5 - 1.0)
            # Foot lifts and arcs forward-upward to prepare for next stance
            progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)  # 0.0 -> 1.0 during swing
            
            # Forward motion: foot moves from rear to front
            # At progress=0: starting from rear stance position
            # At progress=1: reaching forward stance position
            foot[0] += self.swing_step_x * (progress - 0.5)
            
            # Upward arc: sinusoidal trajectory for smooth lift and placement
            # Peak at progress=0.5
            swing_arc = np.sin(np.pi * progress)
            foot[2] += self.swing_height * swing_arc
            
            # Net upward displacement to account for base climb
            # Foot must reach higher to match escalator climb
            foot[2] += self.swing_step_z * progress
        
        return foot