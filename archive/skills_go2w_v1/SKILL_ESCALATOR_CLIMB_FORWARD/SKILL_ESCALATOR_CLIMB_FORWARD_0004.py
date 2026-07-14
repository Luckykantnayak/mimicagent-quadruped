from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ESCALATOR_CLIMB_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Escalator climbing gait with diagonal leg pairing and continuous forward+upward base motion.

    - Base moves with constant forward velocity (vx) and constant upward velocity (vz)
    - Diagonal leg pairs alternate: FL+RR swing early, FR+RL swing later
    - Duty cycle 0.75 with proper phase offset ensures continuous ground contact
    - Stance legs move backward AND downward in body frame to counteract base motion
    - Swing legs execute forward arc with vertical clearance from depressed stance position
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 0.8  # Escalator climbing is slower than trot

        # Base foot positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Diagonal pairing phase offsets
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0  # Group 1
            else:  # FR or RL
                self.phase_offsets[leg] = 0.35  # Group 2 offset for contact overlap

        # Gait parameters with reduced range to avoid joint limits
        self.duty_cycle = 0.78  # Increased for better contact overlap
        self.step_length = 0.12  # Reduced to decrease reach envelope
        self.step_height = 0.08  # Reduced to decrease vertical range

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Escalator climb parameters - reduced for joint limit safety
        self.vx = 0.25  # Forward velocity (m/s)
        self.vz = 0.10  # Upward velocity (m/s) - reduced from 0.15
        
        # Calculate required vertical compensation during stance
        # During full stance phase duration, base rises by: vz * (duty_cycle / freq)
        stance_duration = self.duty_cycle / self.freq
        self.stance_descent = self.vz * stance_duration  # Linear descent rate to match base rise
        
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
        Stance phase: foot moves backward AND downward in body frame to maintain contact
        """
        # Compute leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()

        # Determine if in swing or stance
        if leg_phase < self.duty_cycle:
            # STANCE PHASE
            # Foot in contact: moves backward and downward in body frame
            # Linear downward motion compensates for upward base velocity to maintain ground contact
            
            stance_progress = leg_phase / self.duty_cycle  # 0 to 1 over stance duration
            
            # Backward motion in body frame: from front to back
            # Use smooth interpolation for horizontal motion to reduce jerk
            t_smooth = 3 * stance_progress * stance_progress - 2 * stance_progress * stance_progress * stance_progress
            foot[0] += self.step_length * (0.5 - t_smooth)
            
            # Downward progression: LINEAR to exactly counteract base climb
            # This produces constant descent velocity matching vz, keeping world z constant
            foot[2] -= self.stance_descent * stance_progress
            
        else:
            # SWING PHASE
            # Foot lifts from depressed stance position, swings forward and upward, returns to nominal
            
            swing_progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)  # 0 to 1
            
            # Forward displacement: move from back to front with smooth interpolation
            t_smooth = 3 * swing_progress * swing_progress - 2 * swing_progress * swing_progress * swing_progress
            foot[0] += self.step_length * (t_smooth - 0.5)
            
            # Vertical trajectory: lift from depressed position, arc upward, return to nominal
            # At swing_progress=0: foot is at z - stance_descent (end of stance)
            # At swing_progress=1: foot returns to z (nominal position for next stance)
            
            # Swing arc with smooth sinusoidal profile for clearance
            swing_angle = np.pi * swing_progress
            clearance = self.step_height * np.sin(swing_angle)
            
            # Vertical motion: rise from depressed stance position back to nominal
            # Linear rise from -stance_descent to 0, plus clearance arc
            vertical_recovery = self.stance_descent * swing_progress
            foot[2] += -self.stance_descent + vertical_recovery + clearance

        return foot