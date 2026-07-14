from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_TROT_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Steady forward trot gait with diagonal leg pairs alternating between stance and swing.
    
    - Diagonal pair 1 (FL, RR): stance [0.0, 0.5], swing [0.5, 1.0]
    - Diagonal pair 2 (FR, RL): swing [0.0, 0.5], stance [0.5, 1.0]
    - Base moves forward at constant velocity with level orientation
    - Swing legs follow arced trajectories in body frame
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 1.0
        
        # Step parameters
        self.step_length = 0.12
        self.step_height = 0.08
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Diagonal trot phase offsets
        # Group 1 (FL, RR): phase 0.0 -> stance at [0.0, 0.5]
        # Group 2 (FR, RL): phase 0.5 -> stance at [0.5, 1.0] (equivalent to swing at [0.0, 0.5])
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

        # Base velocity parameters
        self.forward_velocity = 0.5
        
        # Duty cycle: 0.5 for trot (50% stance, 50% swing per leg)
        self.duty = 0.5

    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward velocity and zero angular rates.
        Maintains level orientation and straight-line forward motion.
        """
        # Constant forward velocity, no lateral or vertical motion
        self.vel_world = np.array([self.forward_velocity, 0.0, 0.0])
        
        # Zero angular rates to maintain level orientation
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
        
        Stance phase [0.0, 0.5]: Foot sweeps rearward in body frame as base moves forward
        Swing phase [0.5, 1.0]: Foot follows forward arc with ground clearance
        """
        # Apply phase offset for diagonal coordination
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()

        if leg_phase < self.duty:
            # Stance phase: foot sweeps rearward in body frame
            # Progress from 0.0 (front) to 1.0 (rear)
            stance_progress = leg_phase / self.duty
            
            # Foot position sweeps from +step_length/2 (forward) to -step_length/2 (rearward)
            foot[0] += self.step_length * (0.5 - stance_progress)
            
        else:
            # Swing phase: foot follows forward arc
            # Progress from 0.0 (lift-off) to 1.0 (touchdown)
            swing_progress = (leg_phase - self.duty) / (1.0 - self.duty)
            
            # Forward displacement: from -step_length/2 (rear) to +step_length/2 (front)
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # Vertical arc: smooth sinusoidal lift with peak at mid-swing
            arc_angle = np.pi * swing_progress
            foot[2] += self.step_height * np.sin(arc_angle)

        return foot