from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_BOUND_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Bound gait with alternating front/rear leg pair support.
    
    - Front legs (FL, FR) stance during phase 0.0–0.5, swing during 0.5–1.0
    - Rear legs (RL, RR) swing during phase 0.0–0.5, stance during 0.5–1.0
    - Base moves forward continuously with oscillating pitch
    - Pitch rate negative during front stance, positive during rear stance
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 3.0
        
        # Gait timing parameters
        self.stance_duration = 0.5
        self.swing_duration = 0.5
        
        # Foot trajectory parameters
        self.step_length = 0.15
        self.step_height = 0.10
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        for leg in self.leg_names:
            width = self.base_feet_pos_body[leg][1]
            print(f"  {leg}: y = {width:.4f} m")
        
        # Phase offsets for bound gait
        # Front legs: stance 0.0-0.5, swing 0.5-1.0
        # Rear legs: swing 0.0-0.5, stance 0.5-1.0
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('F'):
                self.phase_offsets[leg] = 0.0
            elif leg.startswith('R'):
                self.phase_offsets[leg] = 0.5
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters
        self.forward_velocity = 1.
        self.pitch_rate_amplitude = 1.5

    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward velocity and alternating pitch rate.
        
        Phase 0.0-0.5: Front stance, nose pitches down (negative pitch rate)
        Phase 0.5-1.0: Rear stance, nose pitches up (positive pitch rate)
        """
        vx = self.forward_velocity
        
        # Pitch rate alternates: negative during front stance, positive during rear stance
        if phase < 0.5:
            pitch_rate = -self.pitch_rate_amplitude
        else:
            pitch_rate = self.pitch_rate_amplitude
        
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
        Compute foot position in body frame for bound gait.
        
        Front legs (FL, FR):
          - Phase 0.0-0.5: stance (foot sweeps backward in body frame)
          - Phase 0.5-1.0: swing (foot lifts and moves forward)
        
        Rear legs (RL, RR):
          - Phase 0.0-0.5: swing (foot lifts and moves forward)
          - Phase 0.5-1.0: stance (foot sweeps backward in body frame)
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this leg is in stance or swing at current leg_phase
        # For legs with offset 0.0 (front): stance is 0.0-0.5, swing is 0.5-1.0
        # For legs with offset 0.5 (rear): after wrapping, stance is 0.0-0.5, swing is 0.5-1.0
        
        if leg_phase < self.stance_duration:
            # Stance phase: foot sweeps backward as body moves forward
            progress = leg_phase / self.stance_duration
            foot[0] += self.step_length * (0.5 - progress)
        else:
            # Swing phase: foot lifts, arcs forward, and descends
            progress = (leg_phase - self.stance_duration) / self.swing_duration
            
            # Forward motion: move from back to front
            foot[0] += self.step_length * (progress - 0.5)
            
            # Vertical motion: arc trajectory using sine for smooth lift and landing
            angle = np.pi * progress
            foot[2] += self.step_height * np.sin(angle)
        
        return foot