from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_BOUND_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Bound gait with front-rear leg pair alternation and continuous forward motion.
    
    Phase 0.0-0.5: Front legs (FL, FR) in stance, rear legs (RL, RR) swing forward
    Phase 0.5-1.0: Rear legs (RL, RR) in stance, front legs (FL, FR) swing forward
    
    Base motion: Continuous forward velocity with gentle pitch oscillation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Gait parameters
        self.step_length = 0.12
        self.step_height = 0.06
        
        # Base foot positions in body frame (use only x,y; z will be enforced during stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base velocity parameters
        self.vx_forward = 0.6
        self.pitch_rate_amp = 0.3  # Reduced from 0.4 to minimize pitch-induced ground conflicts

    def update_base_motion(self, phase, dt):
        """
        Update base with continuous forward velocity and pitch oscillation.
        
        Pitch rate oscillates:
        - Phase 0.0-0.5: negative (nose down) as front legs push
        - Phase 0.5-1.0: positive (nose up) as rear legs push
        """
        vx = self.vx_forward
        
        # Pitch rate: sinusoidal oscillation with period matching gait cycle
        # sin(2π*phase) gives smooth transition: negative first half, positive second half
        pitch_rate = self.pitch_rate_amp * np.sin(2 * np.pi * phase)
        
        self.vel_world = np.array([vx, 0.0, 0.0])
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
        Compute foot position in body frame based on leg group and phase.
        
        Front legs (FL, FR): stance [0.0, 0.5], swing [0.5, 1.0]
        Rear legs (RL, RR): swing [0.0, 0.5], stance [0.5, 1.0]
        
        Key fix: Enforce z=0 (ground contact) during all stance phases
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg group
        is_front = leg_name.startswith('F')
        
        if is_front:
            # Front legs: stance first half, swing second half
            if phase < 0.5:
                # Stance phase: sweep rearward, enforce ground contact
                stance_progress = phase / 0.5
                foot[0] += self.step_length * (0.5 - stance_progress)
                foot[2] = 0.0  # Enforce ground contact during stance
            else:
                # Swing phase: arc forward
                swing_progress = (phase - 0.5) / 0.5
                angle = np.pi * swing_progress
                foot[0] += self.step_length * (swing_progress - 0.5)
                foot[2] = self.step_height * np.sin(angle)  # Arc from z=0 to z=0
        else:
            # Rear legs: swing first half, stance second half
            if phase < 0.5:
                # Swing phase: arc forward
                swing_progress = phase / 0.5
                angle = np.pi * swing_progress
                foot[0] += self.step_length * (swing_progress - 0.5)
                foot[2] = self.step_height * np.sin(angle)  # Arc from z=0 to z=0
            else:
                # Stance phase: sweep rearward, enforce ground contact
                stance_progress = (phase - 0.5) / 0.5
                foot[0] += self.step_length * (0.5 - stance_progress)
                foot[2] = 0.0  # Enforce ground contact during stance
        
        return foot

    def get_desired_motion(self):
        phase = (self.t * self.freq) % 1.0
        dt = 1.0 / 60.0
        
        self.update_base_motion(phase, dt)
        
        foot_targets_body = {
            leg: self.compute_foot_position_body_frame(leg, phase)
            for leg in self.leg_names
        }
        
        self.t += dt
        
        return self.root_pos, self.root_quat, self.vel_world, self.omega_world, foot_targets_body