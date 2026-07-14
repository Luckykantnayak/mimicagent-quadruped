from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ESCALATOR_CLIMB_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Escalator climbing gait with diagonal leg pairing and continuous forward+upward base motion.

    - Base moves with constant forward velocity (vx) and constant upward velocity (vz)
    - Diagonal leg pairs alternate: FL+RR swing [0.0-0.35], FR+RL swing [0.5-0.85]
    - Duty cycle 0.7 ensures continuous ground contact with overlapping stance phases
    - Stance legs move backward in body frame to counteract base motion
    - Swing legs execute forward arc with vertical clearance
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 0.8  # Escalator climbing is slower than trot

        # Base foot positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Diagonal pairing phase offsets
        # Group 1: FL+RR swing early in cycle
        # Group 2: FR+RL swing mid-cycle
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0  # Group 1
            else:  # FR or RL
                self.phase_offsets[leg] = 0.5  # Group 2

        # Gait parameters
        self.duty_cycle = 0.7  # 70% stance, 30% swing - ensures continuous ground contact
        self.step_length = 0.15  # Forward stride length in body frame
        self.step_height = 0.12  # Peak swing height above base position

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Escalator climb parameters
        self.vx = 0.3  # Forward velocity (m/s)
        self.vz = 0.15  # Upward velocity (m/s)
        
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
        Stance phase: foot moves backward in body frame (stays in place in world frame)
        """
        # Compute leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()

        # Determine if in swing or stance
        if leg_phase < self.duty_cycle:
            # STANCE PHASE [0.0 - 0.7]
            # Foot in contact: moves backward in body frame to counteract forward base motion
            # Vertical position remains near base height (no downward compensation needed)
            
            stance_progress = leg_phase / self.duty_cycle  # 0 to 1 over stance duration
            
            # Backward motion in body frame: from front to back
            # Start at +step_length/2, end at -step_length/2
            foot[0] += self.step_length * (0.5 - stance_progress)
            
            # Minimal vertical modulation for smooth ground contact
            # Small sinusoidal dip to ensure contact throughout stance
            contact_assurance = 0.01 * np.sin(np.pi * stance_progress)
            foot[2] -= contact_assurance
            
        else:
            # SWING PHASE [0.7 - 1.0]
            # Foot lifts, swings forward and upward, then descends
            
            swing_progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)  # 0 to 1
            
            # Forward displacement: move from back to front
            # Start at -step_length/2, end at +step_length/2
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # Vertical clearance: sinusoidal arc with smooth entry and exit
            # Peak at swing_progress=0.5 (mid-swing)
            swing_angle = np.pi * swing_progress
            foot[2] += self.step_height * np.sin(swing_angle)

        return foot