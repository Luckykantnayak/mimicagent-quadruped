from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_TROT_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Trot gait with constant forward velocity.
    
    Diagonal leg pairs alternate between stance and swing phases:
    - Group 1 (FL, RR): stance [0.0, 0.5], swing [0.5, 1.0]
    - Group 2 (FR, RL): swing [0.0, 0.5], stance [0.5, 1.0]
    
    Base moves forward at constant velocity with zero angular rates.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Gait parameters
        self.step_length = 0.15  # Total fore-aft excursion in body frame
        self.step_height = 0.06  # Peak swing clearance height
        self.duty_cycle = 0.5    # Each leg spends 50% in stance, 50% in swing
        
        # Base foot positions (neutral stance in body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for diagonal trot coordination
        # FL and RR are synchronized (offset 0.0)
        # FR and RL are synchronized and anti-phase to FL/RR (offset 0.5)
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0  # Group 1: stance first half
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.phase_offsets[leg] = 0.5  # Group 2: swing first half
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Constant forward velocity (tuned so integrated displacement matches step length)
        # Over half cycle (duty_cycle period), base moves forward by step_length
        # vx * (0.5 / freq) = step_length => vx = step_length * freq / 0.5
        self.vx_forward = self.step_length * self.freq / 0.5

    def update_base_motion(self, phase, dt):
        """
        Update base pose using constant forward velocity and zero angular rates.
        """
        # Constant forward velocity in x, zero lateral and vertical
        self.vel_world = np.array([self.vx_forward, 0.0, 0.0])
        
        # Zero angular rates to maintain level orientation
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        # Integrate pose in world frame
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
        
        Stance phase (leg-specific phase 0.0 to 0.5):
            Foot moves rearward in body frame as base advances forward.
            Linear rearward motion from +step_length/2 to -step_length/2 in x.
        
        Swing phase (leg-specific phase 0.5 to 1.0):
            Foot arcs forward and upward from rear position to forward position.
            Smooth arc trajectory with sinusoidal height profile.
        """
        # Apply phase offset for this leg to get leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_phase < self.duty_cycle:
            # Stance phase: foot slides rearward in body frame
            # At leg_phase=0.0: foot at forward position (+step_length/2)
            # At leg_phase=0.5: foot at rear position (-step_length/2)
            stance_progress = leg_phase / self.duty_cycle  # 0.0 to 1.0
            foot[0] += self.step_length * (0.5 - stance_progress)
            # z remains at ground level (no change from base position)
        else:
            # Swing phase: foot arcs forward and upward
            # At leg_phase=0.5: lift from rear position (-step_length/2)
            # At leg_phase=0.75: peak height at mid-swing
            # At leg_phase=1.0: land at forward position (+step_length/2)
            swing_progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)  # 0.0 to 1.0
            
            # Forward motion during swing: from -step_length/2 to +step_length/2
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # Upward arc: sinusoidal profile for smooth lift and landing
            arc_angle = np.pi * swing_progress  # 0 to π
            foot[2] += self.step_height * np.sin(arc_angle)
        
        return foot