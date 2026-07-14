from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ESCALATOR_CLIMB_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Escalator climbing motion with continuous forward progression and climbing appearance.
    
    - Base moves with constant forward velocity; vertical position stable with minimal oscillation
    - Diagonal trot gait: FL+RR alternate with FR+RL
    - 50% duty cycle per leg group
    - Stance phase: foot sweeps backward in body frame (creates climbing appearance)
    - Swing phase: foot arcs forward with continuity enforcement
    - No flight phase: always two feet in contact
    - Minimal vertical foot motion to prevent ground penetration
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
        
        # Stance phase foot motion (backward sweep in body frame, minimal vertical motion)
        self.stance_sweep_x = 0.10  # Forward-backward sweep range during stance
        self.stance_sweep_z = 0.0   # No downward sweep to prevent ground penetration
        
        # Swing phase must close the loop: swing displacement = stance displacement
        self.swing_step_x = 0.10   # Must equal stance_sweep_x for continuity
        self.swing_step_z = 0.0    # Must equal stance_sweep_z for continuity
        self.swing_height = 0.05   # Peak clearance height (adequate for smooth swing)
        
        # Base velocity parameters (world frame)
        self.vx_climb = 0.35   # Forward velocity (m/s) - creates forward progression
        self.vz_base_oscillation_amp = 0.005  # Minimal vertical oscillation for natural appearance
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Initialize velocity commands
        self.vel_world = np.array([self.vx_climb, 0.0, 0.0])
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Base moves forward with constant velocity.
        Minimal vertical oscillation to maintain natural appearance without causing penetration.
        Body orientation remains stable.
        """
        # Constant forward velocity
        vx = self.vx_climb
        
        # Minimal sinusoidal vertical oscillation synchronized with gait
        # Creates subtle natural climbing appearance without height drift
        vz = self.vz_base_oscillation_amp * 2.0 * np.pi * np.cos(2.0 * np.pi * phase)
        
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
        Compute foot position in body frame with enforced stance-swing continuity.
        
        Stance phase (leg_phase 0.0-0.5):
          - Foot sweeps backward (negative x) in body frame
          - Minimal vertical motion to prevent ground penetration
          - Creates climbing appearance through horizontal motion as base moves forward
        
        Swing phase (leg_phase 0.5-1.0):
          - Foot lifts and arcs forward
          - MUST begin exactly where stance ended and end where stance begins
          - Enforces kinematic loop closure to prevent flight phase
        """
        # Apply phase offset for diagonal coordination
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_phase < self.duty_cycle:
            # STANCE PHASE (0.0 - 0.5)
            # Foot sweeps from forward position to rear position
            progress = leg_phase / self.duty_cycle  # 0.0 -> 1.0 during stance
            
            # X motion: forward to rear (creates backward sweep as base moves forward)
            # progress=0: foot at +stance_sweep_x/2 (forward)
            # progress=1: foot at -stance_sweep_x/2 (rear)
            x_offset = self.stance_sweep_x * 0.5 * (1.0 - 2.0 * progress)
            
            # Z motion: remain at base height (no downward sweep to prevent penetration)
            z_offset = 0.0
            
            foot[0] += x_offset
            foot[2] += z_offset
            
        else:
            # SWING PHASE (0.5 - 1.0)
            # Foot arcs from rear position (end of stance) to forward position (start of stance)
            progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)  # 0.0 -> 1.0 during swing
            
            # Compute stance end position (where swing must begin)
            stance_end_x = -self.stance_sweep_x * 0.5
            stance_end_z = 0.0  # No vertical offset from base position
            
            # Compute stance start position (where swing must end)
            stance_start_x = self.stance_sweep_x * 0.5
            stance_start_z = 0.0  # No vertical offset from base position
            
            # Linear interpolation from stance end to stance start (baseline trajectory)
            x_baseline = stance_end_x + (stance_start_x - stance_end_x) * progress
            z_baseline = stance_end_z + (stance_start_z - stance_end_z) * progress
            
            # Add smooth vertical arc for swing clearance
            # Use smoothed sinusoidal arc for continuous velocity
            arc = np.sin(np.pi * progress)
            z_arc = self.swing_height * arc
            
            # Apply trajectory
            foot[0] += x_baseline
            foot[2] += z_baseline + z_arc
        
        return foot