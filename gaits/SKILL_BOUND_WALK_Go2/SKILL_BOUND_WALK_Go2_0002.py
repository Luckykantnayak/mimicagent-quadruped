from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_BOUND_WALK_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Bound gait with alternating front and rear leg pair support.
    
    - Front legs (FL, FR) and rear legs (RL, RR) operate as synchronized pairs
    - Front pair in stance during phase [0.0, 0.5], rear pair swings
    - Rear pair in stance during phase [0.5, 1.0], front pair swings
    - Continuous forward velocity with mild pitch oscillation
    - At least two feet always in contact (either front or rear pair)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Gait parameters
        self.step_length = 0.12  # Forward reach during swing
        self.step_height = 0.06  # Swing foot clearance
        self.stance_sweep = 0.12  # Rearward foot motion during stance in body frame
        
        # Base foot positions (neutral stance in body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets: front pair synchronized at 0.0, rear pair at 0.5
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('F'):  # FL, FR
                self.phase_offsets[leg] = 0.0
            elif leg.startswith('R'):  # RL, RR
                self.phase_offsets[leg] = 0.5
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters
        self.vx_forward = 0.4  # Steady forward velocity
        self.pitch_rate_amp = 0.3  # Pitch oscillation amplitude (rad/s)
        
    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward velocity and sinusoidal pitch rate.
        
        - Forward velocity: constant positive
        - Pitch rate: positive during front stance (phase 0-0.5), negative during rear stance (phase 0.5-1.0)
        - This creates mild pitch oscillation synchronized with weight transfer
        """
        # Constant forward velocity
        vx = self.vx_forward
        
        # Sinusoidal pitch rate: positive when front legs push (phase 0-0.5), negative when rear legs push
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
        Compute foot position in body frame based on phase and contact state.
        
        For front legs (FL, FR):
          - phase [0.0, 0.5]: stance (sweep rearward in body frame)
          - phase [0.5, 1.0]: swing (arc forward and upward)
        
        For rear legs (RL, RR):
          - phase [0.0, 0.5]: swing (arc forward and upward)
          - phase [0.5, 1.0]: stance (sweep rearward in body frame)
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front_leg = leg_name.startswith('F')
        
        if is_front_leg:
            # Front legs: stance [0.0, 0.5], swing [0.5, 1.0]
            if leg_phase < 0.5:
                # Stance phase: foot sweeps rearward as body moves forward
                progress = leg_phase / 0.5  # 0 -> 1 over stance
                # Start forward, end rearward
                foot[0] += self.stance_sweep * (0.5 - progress)
            else:
                # Swing phase: foot arcs forward with clearance
                progress = (leg_phase - 0.5) / 0.5  # 0 -> 1 over swing
                # Arc trajectory
                foot[0] += self.step_length * (progress - 0.5)
                swing_angle = np.pi * progress
                foot[2] += self.step_height * np.sin(swing_angle)
        else:
            # Rear legs: swing [0.0, 0.5], stance [0.5, 1.0]
            if leg_phase < 0.5:
                # Swing phase: foot arcs forward with clearance
                progress = leg_phase / 0.5  # 0 -> 1 over swing
                # Arc trajectory
                foot[0] += self.step_length * (progress - 0.5)
                swing_angle = np.pi * progress
                foot[2] += self.step_height * np.sin(swing_angle)
            else:
                # Stance phase: foot sweeps rearward as body moves forward
                progress = (leg_phase - 0.5) / 0.5  # 0 -> 1 over stance
                # Start forward, end rearward
                foot[0] += self.stance_sweep * (0.5 - progress)
        
        return foot