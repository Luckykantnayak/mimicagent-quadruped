from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_HOURGLASS_EXPANSION_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Hourglass Expansion Walk: A forward walking gait where all four legs
    rhythmically converge toward the body centerline then expand outward,
    creating an hourglass-shaped footprint pattern. The base height oscillates
    inversely with stance width: rising when legs narrow, lowering when legs widen.
    Forward progression is continuous throughout the cycle.
    
    All four feet remain in contact throughout the entire cycle.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slow gait frequency for smooth hourglass visualization
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Hourglass motion parameters
        self.lateral_amplitude = 0.12  # Amplitude of lateral (y-axis) oscillation
        self.vertical_amplitude = 0.06  # Amplitude of base height oscillation
        self.forward_step_length = 0.08  # Forward displacement per cycle
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity parameters
        self.base_forward_velocity = 0.15  # Base forward velocity magnitude
        
    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent vertical and forward velocities.
        
        Base rises during convergence (phase 0.0-0.4) and descends during expansion (phase 0.4-0.8).
        Forward velocity is maintained throughout with slight modulation for realism.
        """
        
        # Forward velocity (continuous with slight variation for natural motion)
        vx = self.base_forward_velocity * (1.0 + 0.2 * np.sin(2 * np.pi * phase))
        
        # Vertical velocity (inverse correlation with stance width)
        # Rise during convergence (0.0-0.4), descend during expansion (0.4-0.8)
        if phase < 0.2:
            # Converging rise phase
            vz = self.vertical_amplitude * 2.0 * np.sin(np.pi * phase / 0.2)
        elif phase < 0.4:
            # Narrow peak phase (transitioning from rise to level)
            vz = self.vertical_amplitude * 2.0 * np.sin(np.pi * (0.4 - phase) / 0.2)
        elif phase < 0.6:
            # Expanding descent phase
            vz = -self.vertical_amplitude * 2.0 * np.sin(np.pi * (phase - 0.4) / 0.2)
        elif phase < 0.8:
            # Wide trough phase (transitioning from descent to level)
            vz = -self.vertical_amplitude * 2.0 * np.sin(np.pi * (0.8 - phase) / 0.2)
        else:
            # Reconverging rise phase
            vz = self.vertical_amplitude * 2.0 * np.sin(np.pi * (phase - 0.8) / 0.2)
        
        # No lateral or angular velocities
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
        Compute foot position in BODY frame with synchronized lateral oscillation
        and continuous forward progression.
        
        All legs move in unison:
        - Lateral: converge inward (0.0-0.4), expand outward (0.4-0.8)
        - Longitudinal: small forward adjustments to maintain forward progression
        - Vertical: remain on ground (z = base_z)
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine lateral direction based on leg position
        # Left legs (FL, RL): positive y is outward
        # Right legs (FR, RR): negative y is outward
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            lateral_sign = 1.0
        else:
            lateral_sign = -1.0
        
        # Lateral oscillation (hourglass pattern)
        # Phase 0.0-0.4: converge inward (reduce lateral offset)
        # Phase 0.4-0.8: expand outward (increase lateral offset)
        # Phase 0.8-1.0: begin reconvergence
        
        # Create smooth sinusoidal lateral modulation
        # At phase 0.2: minimum width (maximum inward)
        # At phase 0.7: maximum width (maximum outward)
        lateral_phase_shift = (phase + 0.3) % 1.0  # Shift to align min at 0.2
        lateral_offset = -self.lateral_amplitude * np.cos(2 * np.pi * lateral_phase_shift)
        
        foot[1] += lateral_sign * lateral_offset
        
        # Forward progression (small cyclic adjustments maintaining net forward motion)
        # Front legs slightly ahead during narrow stance, rear legs catch up during wide stance
        if leg_name.startswith('FL') or leg_name.startswith('FR'):
            # Front legs
            forward_adjustment = self.forward_step_length * (0.5 * np.sin(2 * np.pi * phase))
        else:
            # Rear legs (phase-shifted for natural gait)
            forward_adjustment = self.forward_step_length * (0.5 * np.sin(2 * np.pi * phase + np.pi))
        
        foot[0] += forward_adjustment
        
        # Vertical position remains constant (ground contact)
        # No modification to foot[2], maintaining initial z position
        
        return foot