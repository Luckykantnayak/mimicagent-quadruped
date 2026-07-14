from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_AUGER_DRILL_DESCENT_MotionGenerator(BaseMotionGenerator):
    """
    Auger drill descent motion: continuous yaw rotation with uniform vertical descent.
    
    - Base continuously rotates about yaw axis (multiple full rotations per cycle)
    - Base descends uniformly from max to min height
    - All four legs maintain continuous ground contact
    - Legs execute synchronized helical trajectories in body frame
    - Creates drill-bit visual effect with spiral ground traces
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for controlled drilling motion
        
        # Base foot positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for 90° intervals (four-fold symmetry)
        self.phase_offsets = {
            leg_names[0]: 0.00,   # FL: 0°
            leg_names[1]: 0.25,   # FR: 90°
            leg_names[2]: 0.50,   # RL: 180°
            leg_names[3]: 0.75,   # RR: 270°
        }
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Drilling motion parameters
        self.yaw_rate = 4.0 * 2 * np.pi  # 4 full rotations per phase cycle
        self.descent_rate = -0.3  # Uniform downward velocity (m/s)
        
        # Helical leg motion parameters
        self.radial_amplitude = 0.05  # Amplitude of radial modulation
        self.vertical_descent = 0.15  # Total vertical descent of feet in body frame
        self.helix_cycles = 2.0  # Number of helical cycles per phase

    def update_base_motion(self, phase, dt):
        """
        Update base with constant yaw rotation and uniform vertical descent.
        """
        # Constant yaw rate for continuous rotation
        yaw_rate = self.yaw_rate
        
        # Constant negative z-velocity for uniform descent
        vz = self.descent_rate
        
        self.vel_world = np.array([0.0, 0.0, vz])
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
        Compute helical foot trajectory in body frame.
        
        Each foot spirals outward and downward, creating a helix:
        - Radial distance modulates sinusoidally
        - Vertical position decreases uniformly
        - Phase offsets create 90° symmetry between legs
        """
        # Apply phase offset for this leg
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute base radial distance and angle in body frame
        base_radius = np.sqrt(foot[0]**2 + foot[1]**2)
        base_angle = np.arctan2(foot[1], foot[0])
        
        # Helical radial modulation: spiral outward and back
        helix_angle = 2 * np.pi * self.helix_cycles * leg_phase
        radial_modulation = self.radial_amplitude * np.sin(helix_angle)
        
        # New radial distance
        new_radius = base_radius + radial_modulation
        
        # Maintain angular position relative to body frame
        # (base rotation handles the spinning)
        foot[0] = new_radius * np.cos(base_angle)
        foot[1] = new_radius * np.sin(base_angle)
        
        # Uniform vertical descent through phase
        foot[2] -= self.vertical_descent * leg_phase
        
        return foot