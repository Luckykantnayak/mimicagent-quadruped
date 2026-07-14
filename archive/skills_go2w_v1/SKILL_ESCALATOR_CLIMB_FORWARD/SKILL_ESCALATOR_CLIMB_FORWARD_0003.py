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
        # Offset chosen to ensure overlap: with duty_cycle 0.75, swing is 0.25 duration
        # Group 1 swings at local 0.75-1.0, Group 2 swings at local 0.75-1.0
        # With 0.5 offset: Group 2 global swing is 0.25-0.5
        # At global 0.75, Group 1 enters swing, Group 2 is at local 0.25 (in stance)
        # At global 0.25, Group 2 enters swing, Group 1 is at local 0.75 (entering swing) - PROBLEM
        # Need offset = 0.375 so Group 2 swing (global 0.125-0.375) doesn't overlap Group 1 swing (global 0.75-1.0)
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0  # Group 1
            else:  # FR or RL
                self.phase_offsets[leg] = 0.375  # Group 2 offset to prevent overlap

        # Gait parameters
        self.duty_cycle = 0.75  # 75% stance, 25% swing - ensures continuous ground contact
        self.step_length = 0.15  # Forward stride length in body frame
        self.step_height = 0.12  # Peak swing height above stance exit position

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Escalator climb parameters
        self.vx = 0.3  # Forward velocity (m/s)
        self.vz = 0.15  # Upward velocity (m/s)
        
        # Calculate required vertical compensation during stance
        # During full stance phase, base rises by: (vz / freq) * duty_cycle
        self.stance_descent = (self.vz / self.freq) * self.duty_cycle  # ~0.14 meters
        
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
            # STANCE PHASE [0.0 - 0.75]
            # Foot in contact: moves backward and downward in body frame
            # Downward motion compensates for upward base velocity to maintain ground contact
            
            stance_progress = leg_phase / self.duty_cycle  # 0 to 1 over stance duration
            
            # Backward motion in body frame: from front to back
            # Start at +step_length/2, end at -step_length/2
            foot[0] += self.step_length * (0.5 - stance_progress)
            
            # Downward progression to counteract base climb
            # Use smooth cubic interpolation for continuous velocity at boundaries
            # At start (progress=0): z_offset = 0
            # At end (progress=1): z_offset = -stance_descent
            # Smooth using: z = -stance_descent * (3*t^2 - 2*t^3) for continuous velocity
            t = stance_progress
            smooth_factor = 3 * t * t - 2 * t * t * t
            foot[2] -= self.stance_descent * smooth_factor
            
        else:
            # SWING PHASE [0.75 - 1.0]
            # Foot lifts from depressed stance position, swings forward and upward, returns to nominal
            
            swing_progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)  # 0 to 1
            
            # Forward displacement: move from back to front
            # Start at -step_length/2, end at +step_length/2
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # Vertical trajectory: lift from depressed position, arc upward, return to nominal
            # At swing_progress=0: foot is at z - stance_descent (end of stance)
            # At swing_progress=0.5: foot is at peak (z + step_height - stance_descent/2)
            # At swing_progress=1: foot returns to z (nominal position for next stance)
            
            # Swing arc with smooth entry/exit using sinusoidal profile
            swing_angle = np.pi * swing_progress
            clearance = self.step_height * np.sin(swing_angle)
            
            # Vertical motion: rise from depressed stance position back to nominal
            # Linear rise from -stance_descent to 0, plus clearance arc
            vertical_recovery = self.stance_descent * swing_progress
            foot[2] += -self.stance_descent + vertical_recovery + clearance

        return foot