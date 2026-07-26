from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_REVERSE_CARTWHEEL_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Reverse cartwheel drift: continuous backward drifting motion combined with
    a full 360-degree roll rotation per cycle.
    
    - Base executes constant roll rate to complete 360° over one phase cycle
    - Backward velocity maintained throughout all phases
    - Right legs (FL, FR) overhead during phases 0.0-0.5
    - Left legs (RL, RR) overhead during phases 0.5-1.0
    - Contact alternates between left and right leg pairs
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for complex cartwheel motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Cartwheel parameters
        self.overhead_extension = 0.25  # How far legs extend overhead (body +Z)
        self.stance_retraction = 0.05   # How much legs retract during stance
        self.lateral_spread = 0.08      # Lateral extension during overhead phase
        
        # Base motion parameters
        self.backward_velocity = -0.8   # Constant backward drift (negative x)
        self.roll_rate = 2 * np.pi * self.freq  # 360° per cycle
        
        # Velocity modulation during phases
        self.vx_peak = -1.0      # Peak backward velocity during inverted phase
        self.vx_stance = -0.6    # Reduced velocity during stance phases
        self.vz_lift = 0.15      # Upward velocity during lift-off
        self.vz_land = -0.15     # Downward velocity during landing
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base with constant roll rate and phase-modulated backward velocity.
        
        Phase 0.0-0.25: backward drift with upward component, initiate roll
        Phase 0.25-0.5: peak backward drift during inverted transition
        Phase 0.5-0.75: backward drift with downward component, continue roll
        Phase 0.75-1.0: sustained backward drift, complete roll to upright
        """
        
        # Constant positive roll rate for 360° rotation per cycle
        roll_rate = self.roll_rate
        
        # Phase-dependent backward velocity and vertical velocity
        if phase < 0.25:
            # Right ascent: initiate roll, slight upward velocity
            vx = self.vx_stance
            vz = self.vz_lift * np.sin(phase / 0.25 * np.pi)
        elif phase < 0.5:
            # Inverted transition: peak backward velocity, minimal vertical
            vx = self.vx_peak
            vz = 0.0
        elif phase < 0.75:
            # Left ascent: sustained drift, slight downward velocity for landing
            vx = self.vx_stance
            vz = self.vz_land * np.sin((phase - 0.5) / 0.25 * np.pi)
        else:
            # Upright return: maintain drift, stabilize vertical
            vx = self.vx_stance
            vz = 0.0
        
        # Set world frame velocities
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
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
        Compute foot trajectory in body frame.
        
        FL, FR (right legs):
          - Phase 0.0-0.5: swing overhead (circular arc toward +Z)
          - Phase 0.5-1.0: descend and stance (push backward)
        
        RL, RR (left legs):
          - Phase 0.0-0.25: stance (push backward)
          - Phase 0.25-1.0: swing overhead and descend
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine if leg is on right side (FL, FR) or left side (RL, RR)
        is_right_leg = leg_name.startswith('F')
        
        if is_right_leg:
            # FL, FR: overhead during 0.0-0.5, stance during 0.5-1.0
            if phase < 0.5:
                # Swing overhead: circular arc from ground to +Z
                swing_progress = phase / 0.5
                arc_angle = swing_progress * np.pi  # 0 to π
                
                # Circular trajectory: start at base, arc to overhead
                foot[0] = base_pos[0] + self.stance_retraction * np.cos(arc_angle)
                foot[1] = base_pos[1] + self.lateral_spread * np.sin(arc_angle)
                foot[2] = base_pos[2] + self.overhead_extension * np.sin(arc_angle)
                
            else:
                # Descend and stance: arc from overhead back to ground, then push
                stance_progress = (phase - 0.5) / 0.5
                
                if stance_progress < 0.5:
                    # Descending arc (0.5-0.75 phase)
                    arc_angle = np.pi * (1.0 - stance_progress * 2.0)  # π to 0
                    foot[0] = base_pos[0] + self.stance_retraction * np.cos(arc_angle)
                    foot[1] = base_pos[1] + self.lateral_spread * np.sin(arc_angle)
                    foot[2] = base_pos[2] + self.overhead_extension * np.sin(arc_angle)
                else:
                    # Stance push (0.75-1.0 phase)
                    push_progress = (stance_progress - 0.5) / 0.5
                    # Push backward in body frame
                    foot[0] = base_pos[0] - self.stance_retraction * (push_progress - 0.5)
                    foot[1] = base_pos[1]
                    foot[2] = base_pos[2]
        
        else:
            # RL, RR: stance during 0.0-0.25, overhead during 0.25-1.0
            if phase < 0.25:
                # Stance push: push backward to generate drift and roll momentum
                push_progress = phase / 0.25
                foot[0] = base_pos[0] - self.stance_retraction * (push_progress - 0.5)
                foot[1] = base_pos[1]
                foot[2] = base_pos[2]
                
            elif phase < 0.75:
                # Swing overhead: circular arc from ground to +Z and hold
                swing_progress = (phase - 0.25) / 0.5
                arc_angle = swing_progress * np.pi  # 0 to π
                
                foot[0] = base_pos[0] + self.stance_retraction * np.cos(arc_angle)
                foot[1] = base_pos[1] - self.lateral_spread * np.sin(arc_angle)
                foot[2] = base_pos[2] + self.overhead_extension * np.sin(arc_angle)
                
            else:
                # Descend: arc from overhead back toward ground for next cycle
                descent_progress = (phase - 0.75) / 0.25
                arc_angle = np.pi * (1.0 - descent_progress)  # π to 0
                
                foot[0] = base_pos[0] + self.stance_retraction * np.cos(arc_angle)
                foot[1] = base_pos[1] - self.lateral_spread * np.sin(arc_angle)
                foot[2] = base_pos[2] + self.overhead_extension * np.sin(arc_angle)
        
        return foot