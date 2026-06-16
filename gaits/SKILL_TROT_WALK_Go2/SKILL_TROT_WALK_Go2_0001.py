from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_TROT_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal trot gait with constant forward velocity.
    
    - Phase 0.0-0.5: FL+RR stance, FR+RL swing
    - Phase 0.5-1.0: FR+RL stance, FL+RR swing
    - Base moves forward at constant velocity
    - Foot trajectories in BODY frame
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Gait parameters
        self.step_length = 0.15  # Forward step length in body frame
        self.step_height = 0.06  # Swing arc height
        self.stance_duty = 0.5   # Each leg stance duration (50% of cycle)
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for diagonal trot coordination
        # Group 1 (FL, RR): stance phase 0.0-0.5, swing phase 0.5-1.0
        # Group 2 (FR, RL): swing phase 0.0-0.5, stance phase 0.5-1.0
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0  # Group 1
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.phase_offsets[leg] = 0.5  # Group 2
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Constant forward velocity
        self.vx_forward = 0.4  # Constant forward velocity (m/s)

    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward velocity.
        No lateral, vertical, or rotational motion.
        """
        # Constant forward velocity in world frame
        self.vel_world = np.array([self.vx_forward, 0.0, 0.0])
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
        Compute foot trajectory in body frame for diagonal trot.
        
        Each leg alternates between stance (0.5 cycle) and swing (0.5 cycle).
        - Stance: foot sweeps backward linearly from +step_length/2 to -step_length/2
        - Swing: foot lifts, arcs forward from -step_length/2 to +step_length/2
        """
        # Apply phase offset for diagonal coordination
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Copy base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is in stance or swing
        if leg_phase < self.stance_duty:
            # Stance phase: foot sweeps backward in body frame
            # Progress from 0 (front) to 1 (rear)
            progress = leg_phase / self.stance_duty
            # Linear sweep from +step_length/2 to -step_length/2
            foot[0] += self.step_length * (0.5 - progress)
            
        else:
            # Swing phase: foot arcs forward with vertical clearance
            # Progress from 0 (rear) to 1 (front)
            progress = (leg_phase - self.stance_duty) / (1.0 - self.stance_duty)
            
            # Horizontal motion: sweep from -step_length/2 to +step_length/2
            foot[0] += self.step_length * (progress - 0.5)
            
            # Vertical arc: smooth parabolic trajectory
            # Peak height at progress = 0.5
            arc_angle = np.pi * progress
            foot[2] += self.step_height * np.sin(arc_angle)
        
        return foot