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
        
        # Base foot positions (used only for x,y radial reference)
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
        
        # Height envelope for base descent - reduced to maintain kinematic feasibility
        self.max_height = 0.28
        self.min_height = 0.16
        
        # Drilling motion parameters
        self.yaw_rate = 4.0 * 2 * np.pi  # 4 full rotations per phase cycle
        
        # Helical leg motion parameters - reduced for workspace safety
        self.radial_amplitude = 0.02  # Reduced radial modulation to stay within workspace
        self.helix_cycles = 2.0  # Number of helical cycles per phase
        
        # Workspace limit for foot body-frame z to prevent over-extension
        self.min_allowable_body_z = -0.30

    def update_base_motion(self, phase, dt):
        """
        Update base with constant yaw rotation and phase-mapped height descent.
        """
        # Constant yaw rate for continuous rotation
        yaw_rate = self.yaw_rate
        
        # Phase-mapped height: linear interpolation from max to min
        target_height = self.max_height - (self.max_height - self.min_height) * phase
        
        # Compute vertical velocity to reach target height smoothly
        current_height = self.root_pos[2]
        height_error = target_height - current_height
        # Use proportional control to smoothly track target height
        vz = height_error * 2.0 * np.pi * self.freq
        
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
        Compute helical foot trajectory in body frame that maintains ground contact.
        
        Feet execute helical radial modulation while staying grounded:
        - Radial distance modulates sinusoidally (creates spiral in world frame)
        - Vertical position set to maintain ground contact within workspace limits
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
        foot[0] = new_radius * np.cos(base_angle)
        foot[1] = new_radius * np.sin(base_angle)
        
        # Enforce ground contact while respecting workspace limits
        # Target: foot_body_z = -base_z for ground contact
        # Constraint: foot_body_z >= min_allowable_body_z to prevent over-extension
        target_body_z = -self.root_pos[2]
        foot[2] = max(target_body_z, self.min_allowable_body_z)
        
        return foot