from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ESCALATOR_CLIMB_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Escalator climbing motion with continuous forward progression and climbing appearance.
    
    - Base moves with constant forward velocity; vertical position stable with small oscillation
    - Diagonal trot gait: FL+RR alternate with FR+RL
    - 50% duty cycle per leg group
    - Stance phase: foot sweeps backward-downward in body frame (creates climbing appearance)
    - Swing phase: foot arcs forward-upward with continuity enforcement
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
        self.stance_sweep_x = 0.10  # Forward-backward sweep range during stance
        self.stance_sweep_z = 0.04  # Vertical sweep range during stance (downward motion for climb illusion)
        
        # Swing phase must close the loop: swing displacement = stance displacement
        self.swing_step_x = 0.10   # Must equal stance_sweep_x for continuity
        self.swing_step_z = 0.04   # Must equal stance_sweep_z for continuity
        self.swing_height = 0.04   # Peak clearance height (reduced for contact maintenance)
        
        # Base velocity parameters (world frame)
        self.vx_climb = 0.35   # Forward velocity (m/s) - creates forward progression
        self.vz_base_oscillation_amp = 0.015  # Small vertical oscillation for natural climbing appearance
        
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
        Vertical motion uses small sinusoidal oscillation (no accumulating drift).
        Body orientation remains stable.
        """
        # Constant forward velocity
        vx = self.vx_climb
        
        # Small sinusoidal vertical oscillation synchronized with gait
        # This creates natural climbing appearance without height drift
        # Phase 0.0-0.5: Group1 stance, slight downward motion
        # Phase 0.5-1.0: Group2 stance, slight upward motion
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
          - Foot sweeps backward (negative x) and downward (negative z) in body frame
          - Creates climbing appearance as base moves forward
        
        Swing phase (leg_phase 0.5-1.0):
          - Foot lifts and arcs forward-upward
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
            
            # Z motion: smooth downward sweep (creates climbing appearance)
            # progress=0: foot at base height
            # progress=1: foot at base height - stance_sweep_z
            z_offset = -self.stance_sweep_z * progress
            
            foot[0] += x_offset
            foot[2] += z_offset
            
        else:
            # SWING PHASE (0.5 - 1.0)
            # Foot arcs from rear position (end of stance) to forward position (start of stance)
            progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)  # 0.0 -> 1.0 during swing
            
            # Compute stance end position (where swing must begin)
            stance_end_x = -self.stance_sweep_x * 0.5
            stance_end_z = -self.stance_sweep_z
            
            # Compute stance start position (where swing must end)
            stance_start_x = self.stance_sweep_x * 0.5
            stance_start_z = 0.0
            
            # Linear interpolation from stance end to stance start (baseline trajectory)
            x_baseline = stance_end_x + (stance_start_x - stance_end_x) * progress
            z_baseline = stance_end_z + (stance_start_z - stance_end_z) * progress
            
            # Add smooth vertical arc for swing clearance
            # Use smoothed arc that reduces more rapidly near touchdown
            if progress < 0.5:
                # Rising phase: smooth acceleration
                arc_progress = 2.0 * progress
                arc = np.sin(0.5 * np.pi * arc_progress)
            else:
                # Falling phase: smooth deceleration with earlier ground approach
                arc_progress = 2.0 * (progress - 0.5)
                arc = np.cos(0.5 * np.pi * arc_progress)
            
            z_arc = self.swing_height * arc
            
            # Apply trajectory
            foot[0] += x_baseline
            foot[2] += z_baseline + z_arc
        
        return foot