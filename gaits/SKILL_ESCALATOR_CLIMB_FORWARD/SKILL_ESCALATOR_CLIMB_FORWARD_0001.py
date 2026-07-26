from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ESCALATOR_CLIMB_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Escalator climbing gait with diagonal leg pairing and continuous forward+upward base motion.

    - Base moves with constant forward velocity (vx) and constant upward velocity (vz)
    - Diagonal leg pairs alternate: FL+RR swing [0.0-0.5], FR+RL swing [0.5-1.0]
    - Stance legs move backward and downward in body frame to counteract base motion
    - Swing legs execute forward arc with enhanced vertical clearance
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 0.8  # Escalator climbing is slower than trot

        # Base foot positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Diagonal pairing phase offsets (trot-like)
        # Group 1: FL+RR swing [0.0-0.5], stance [0.5-1.0]
        # Group 2: FR+RL stance [0.0-0.5], swing [0.5-1.0]
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0  # Group 1
            else:  # FR or RL
                self.phase_offsets[leg] = 0.5  # Group 2

        # Gait parameters
        self.duty_cycle = 0.5  # Swing duration is 50% of cycle
        self.step_length = 0.15  # Forward stride length in body frame
        self.step_height = 0.12  # Peak swing height above base position

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Escalator climb parameters
        self.vx = 0.3  # Forward velocity (m/s)
        self.vz = 0.15  # Upward velocity (m/s) - creates ~26.5 degree climb angle
        
        # Velocity commands (constant throughout motion)
        self.vel_world = np.array([self.vx, 0.0, self.vz])
        self.omega_world = np.zeros(3)  # No rotation

    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward and upward velocity.
        No angular rates - body orientation remains fixed.
        """
        # Constant velocity commands throughout entire phase cycle
        self.vel_world = np.array([self.vx, 0.0, self.vz])
        self.omega_world = np.zeros(3)

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
        Compute foot position in body frame based on swing/stance phase.
        
        Swing phase: foot executes forward arc with vertical clearance
        Stance phase: foot moves backward and downward relative to body
        """
        # Compute leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()

        # Determine if in swing or stance
        if leg_phase < self.duty_cycle:
            # SWING PHASE [0.0 - 0.5]
            # Foot lifts, swings forward and upward, then descends
            
            progress = leg_phase / self.duty_cycle  # 0 to 1 over swing duration
            
            # Forward displacement: move from back to front
            # Start at -step_length/2, end at +step_length/2
            foot[0] += self.step_length * (progress - 0.5)
            
            # Vertical clearance: sinusoidal arc
            # Peak at progress=0.5 (mid-swing)
            swing_angle = np.pi * progress
            foot[2] += self.step_height * np.sin(swing_angle)
            
        else:
            # STANCE PHASE [0.5 - 1.0]
            # Foot in contact: moves backward and downward relative to body
            # to counteract forward and upward base motion
            
            stance_progress = (leg_phase - self.duty_cycle) / self.duty_cycle  # 0 to 1
            
            # Backward motion in body frame: from front to back
            # Start at +step_length/2, end at -step_length/2
            foot[0] += self.step_length * (0.5 - stance_progress)
            
            # Downward motion in body frame to accommodate rising base
            # Since base rises at vz, foot must descend relative to body
            # Over stance duration T_stance = (1/freq) * duty_cycle
            T_stance = (1.0 / self.freq) * self.duty_cycle
            z_descent = self.vz * T_stance
            foot[2] -= z_descent * stance_progress

        return foot