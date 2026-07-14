from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ASYNC_LISSAJOUS_CRAWL_MotionGenerator(BaseMotionGenerator):
    """
    Asynchronous crawl with four independent leg trajectories.
    
    Each leg traces a unique curve:
    - FL: forward ellipse
    - FR: lateral figure-eight
    - RL: vertical lift-and-place arc
    - RR: diagonal backward sweep
    
    Base motion integrates asynchronous ground forces to produce
    forward velocity and mild yaw oscillation.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.4
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Per-leg trajectory parameters
        # FL: forward ellipse
        self.fl_ellipse_major = 0.12
        self.fl_ellipse_minor = 0.05
        self.fl_swing_height = 0.15
        
        # FR: lateral figure-eight
        self.fr_fig8_lateral_amp = 0.08
        self.fr_fig8_forward_amp = 0.10
        self.fr_swing_height = 0.10
        
        # RL: vertical arc
        self.rl_vertical_height = 0.09
        self.rl_forward_drift = 0.03
        
        # RR: diagonal sweep
        self.rr_diagonal_forward = 0.14
        self.rr_diagonal_lateral = 0.07
        self.rr_swing_height = 0.06
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base velocity parameters
        self.vx_base = 0.25
        self.yaw_amp = 0.4
        self.yaw_freq = 0.6
        self.vy_amp = 0.05
        self.vy_freq = 0.8

    def update_base_motion(self, phase, dt):
        """
        Base moves forward with oscillating yaw and minor lateral drift.
        Velocity profile simulates integrated asynchronous leg forces.
        """
        # Forward velocity: steady with minor phase modulation
        vx = self.vx_base * (1.0 + 0.15 * np.sin(2 * np.pi * phase))
        
        # Lateral velocity: small oscillation from asymmetric leg forces
        vy = self.vy_amp * np.sin(2 * np.pi * self.vy_freq * phase)
        
        # Yaw rate: oscillates as legs produce asymmetric torques
        # Positive in [0,0.25] and [0.75,1.0], negative in [0.25,0.75]
        yaw_rate = self.yaw_amp * np.sin(2 * np.pi * (phase + 0.25))
        
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Each leg follows a unique trajectory shape with intrinsic phase offsets.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_name.startswith('FL'):
            return self._compute_fl_ellipse(foot, phase)
        elif leg_name.startswith('FR'):
            return self._compute_fr_figure_eight(foot, phase)
        elif leg_name.startswith('RL'):
            return self._compute_rl_vertical_arc(foot, phase)
        elif leg_name.startswith('RR'):
            return self._compute_rr_diagonal_sweep(foot, phase)
        else:
            return foot

    def _compute_fl_ellipse(self, foot_base, phase):
        """
        FL traces a forward-oriented ellipse.
        Stance: [0.0, 0.4] and [0.6, 1.0]
        Swing: [0.4, 0.6]
        """
        foot = foot_base.copy()
        
        # Ellipse parameterization: x = a*cos(theta), z = b*sin(theta)
        # Map phase to ellipse angle with intrinsic offset
        theta = 2 * np.pi * phase
        
        # Horizontal displacement (forward/backward in body frame)
        foot[0] += self.fl_ellipse_major * np.cos(theta)
        
        # Vertical component: only positive during swing phase [0.4, 0.6]
        if 0.4 <= phase < 0.6:
            swing_progress = (phase - 0.4) / 0.2
            swing_angle = np.pi * swing_progress
            foot[2] += self.fl_swing_height * np.sin(swing_angle)
        
        return foot

    def _compute_fr_figure_eight(self, foot_base, phase):
        """
        FR traces a lateral figure-eight (Lissajous curve).
        Stance: [0.0, 0.3] and [0.5, 0.8]
        Swing: [0.3, 0.5] and [0.8, 1.0]
        
        Parametric form:
        x(t) = A * sin(omega*t)
        y(t) = B * sin(2*omega*t)
        Creates a figure-eight in x-y plane when B = 2A (approx)
        """
        foot = foot_base.copy()
        
        # Lissajous parameters
        theta = 2 * np.pi * phase
        
        # Forward component (single frequency)
        foot[0] += self.fr_fig8_forward_amp * np.sin(theta)
        
        # Lateral component (double frequency for figure-eight)
        foot[1] += self.fr_fig8_lateral_amp * np.sin(2 * theta)
        
        # Vertical lift during crossover transitions
        # Swing phases: [0.3, 0.5] (crossover) and [0.8, 1.0] (return)
        if 0.3 <= phase < 0.5:
            swing_progress = (phase - 0.3) / 0.2
            swing_angle = np.pi * swing_progress
            foot[2] += self.fr_swing_height * np.sin(swing_angle)
        elif 0.8 <= phase < 1.0:
            swing_progress = (phase - 0.8) / 0.2
            swing_angle = np.pi * swing_progress
            foot[2] += self.fr_swing_height * 0.7 * np.sin(swing_angle)
        
        return foot

    def _compute_rl_vertical_arc(self, foot_base, phase):
        """
        RL performs vertical lift-and-place with minimal horizontal travel.
        Swing: [0.0, 0.2] and [0.7, 1.0]
        Stance: [0.2, 0.7]
        """
        foot = foot_base.copy()
        
        # Minor forward drift during stance to assist propulsion
        if 0.2 <= phase < 0.7:
            stance_progress = (phase - 0.2) / 0.5
            foot[0] -= self.rl_forward_drift * (stance_progress - 0.5)
        
        # Vertical lift in two swing phases
        if 0.0 <= phase < 0.2:
            # First lift
            lift_progress = phase / 0.2
            lift_angle = np.pi * lift_progress
            foot[2] += self.rl_vertical_height * np.sin(lift_angle)
            # Slight forward reposition during swing
            foot[0] += self.rl_forward_drift * 0.3 * lift_progress
        elif 0.7 <= phase < 1.0:
            # Second lift
            lift_progress = (phase - 0.7) / 0.3
            lift_angle = np.pi * lift_progress
            foot[2] += self.rl_vertical_height * np.sin(lift_angle)
            foot[0] += self.rl_forward_drift * 0.3 * lift_progress
        
        return foot

    def _compute_rr_diagonal_sweep(self, foot_base, phase):
        """
        RR traces a long diagonal sweep (rearward and lateral).
        Stance: [0.0, 0.5] and [0.65, 1.0]
        Swing: [0.5, 0.65]
        """
        foot = foot_base.copy()
        
        # Map phase to diagonal trajectory
        # Diagonal vector: rearward (negative x) and lateral outward (positive y for RR)
        
        if phase < 0.5:
            # First stance: sweep from front to back
            sweep_progress = phase / 0.5
            foot[0] += self.rr_diagonal_forward * (0.5 - sweep_progress)
            foot[1] += self.rr_diagonal_lateral * (0.5 - sweep_progress)
        elif 0.5 <= phase < 0.65:
            # Swing: lift and return to front
            swing_progress = (phase - 0.5) / 0.15
            swing_angle = np.pi * swing_progress
            foot[2] += self.rr_swing_height * np.sin(swing_angle)
            # Arc forward and inward during swing
            foot[0] += self.rr_diagonal_forward * (-0.5 + swing_progress)
            foot[1] += self.rr_diagonal_lateral * (-0.5 + swing_progress)
        else:
            # Second stance: continue diagonal sweep
            sweep_progress = (phase - 0.65) / 0.35
            foot[0] += self.rr_diagonal_forward * (0.5 - sweep_progress)
            foot[1] += self.rr_diagonal_lateral * (0.5 - sweep_progress)
        
        return foot