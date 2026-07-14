from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_BOUND_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Bound gait with alternating front and rear leg pairs.
    
    - Front legs (FL, FR) stance [0.0, 0.5], swing [0.5, 1.0]
    - Rear legs (RL, RR) swing [0.0, 0.5], stance [0.5, 1.0]
    - Continuous forward velocity with mild pitch oscillation
    - Roll remains near zero due to symmetric left-right pairing
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 1.0
        
        # Gait parameters
        self.duty_cycle = 0.5  # Each leg pair in stance for 50% of cycle
        self.step_length = 0.15  # Forward step distance in body frame
        self.step_height = 0.10  # Swing clearance height
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Phase offsets for bound gait coordination
        # Front legs: stance [0.0, 0.5], swing [0.5, 1.0]
        # Rear legs: swing [0.0, 0.5], stance [0.5, 1.0]
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('F'):  # Front legs
                self.phase_offsets[leg] = 0.0
            else:  # Rear legs
                self.phase_offsets[leg] = 0.5

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Base motion parameters
        self.vx_forward = 0.6  # Steady forward velocity
        self.pitch_amplitude = 0.4  # Pitch oscillation amplitude (rad/s)
        self.vz_amplitude = 0.08  # Vertical velocity amplitude
        
    def update_base_motion(self, phase, dt):
        """
        Update base with steady forward velocity and pitch oscillation.
        
        Pitch rate oscillates to create rocking motion:
        - phase [0.0, 0.25]: pitch down (front legs pushing)
        - phase [0.25, 0.5]: pitch up (transition to rear)
        - phase [0.5, 0.75]: pitch up (rear legs pushing)
        - phase [0.75, 1.0]: pitch down (transition to front)
        """
        
        # Steady forward velocity
        vx = self.vx_forward
        
        # Vertical velocity oscillation (mild)
        vz = self.vz_amplitude * np.sin(2 * np.pi * phase)
        
        # Pitch rate oscillation
        # Use cosine to get: negative [0, 0.5], positive [0.5, 1.0]
        pitch_rate = self.pitch_amplitude * np.cos(2 * np.pi * phase)
        
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])

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
        
        Stance phase: foot sweeps backward relative to body
        Swing phase: foot lifts, arcs forward, descends
        """
        
        # Get leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if in stance or swing
        in_stance = leg_phase < self.duty_cycle
        
        if in_stance:
            # Stance phase: foot sweeps backward as body moves forward
            # Progress from 0 (front of stance) to 1 (back of stance)
            progress = leg_phase / self.duty_cycle
            
            # Sweep backward: starts ahead, ends behind
            foot[0] += self.step_length * (0.5 - progress)
            
        else:
            # Swing phase: lift, arc forward, descend
            # Progress from 0 (start of swing) to 1 (end of swing)
            progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)
            
            # Forward motion: starts behind, ends ahead
            foot[0] += self.step_length * (progress - 0.5)
            
            # Vertical arc: sine wave for smooth lift and descent
            swing_angle = np.pi * progress
            foot[2] += self.step_height * np.sin(swing_angle)
        
        return foot