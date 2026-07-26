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
        
        # Base foot positions (used for radial reference and initial angles)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Store initial angular positions in body frame for each foot
        self.initial_foot_angles = {}
        for leg in leg_names:
            foot = self.base_feet_pos_body[leg]
            self.initial_foot_angles[leg] = np.arctan2(foot[1], foot[0])
        
        # Phase offsets for 90° intervals (four-fold symmetry)
        self.phase_offsets = {
            leg_names[0]: 0.00,   # FL: 0°
            leg_names[1]: 0.25,   # FR: 90°
            leg_names[2]: 0.50,   # RL: 180°
            leg_names[3]: 0.75,   # RR: 270°
        }
        
        # Height envelope for base descent - tuned for kinematic feasibility
        self.max_height = 0.28
        self.min_height = 0.16
        
        # Base state - initialize at max_height to avoid startup transient
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.max_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Drilling motion parameters - reduced rotation rate for joint velocity limits
        self.yaw_rate = 1.5 * 2 * np.pi  # 1.5 full rotations per phase cycle
        
        # Helical leg motion parameters
        self.radial_amplitude = 0.015  # Reduced radial modulation amplitude
        self.helix_cycles = 2.0  # Number of helical cycles per phase
        
        # Workspace limit for foot body-frame z - adjusted for height range
        self.min_allowable_body_z = -0.32

    def get_yaw_from_quat(self, quat):
        """
        Extract yaw angle from quaternion [w, x, y, z].
        """
        w, x, y, z = quat
        # Yaw (rotation around z-axis)
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

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
        - Angular position compensates for body yaw to maintain world-frame position
        - Vertical position set to maintain ground contact within workspace limits
        - Phase offsets create 90° symmetry between legs
        """
        # Apply phase offset for this leg
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Get initial foot position for radial reference
        foot = self.base_feet_pos_body[leg_name].copy()
        base_radius = np.sqrt(foot[0]**2 + foot[1]**2)
        
        # Helical radial modulation: spiral outward and back
        helix_angle = 2 * np.pi * self.helix_cycles * leg_phase
        radial_modulation = self.radial_amplitude * np.sin(helix_angle)
        new_radius = base_radius + radial_modulation
        
        # Get current body yaw and compute compensated angular position
        # to maintain world-frame position as body rotates
        current_yaw = self.get_yaw_from_quat(self.root_quat)
        initial_angle = self.initial_foot_angles[leg_name]
        
        # Counter-rotate in body frame to maintain world-frame angular position
        body_frame_angle = initial_angle - current_yaw
        
        # Set foot position with yaw-compensated angle
        foot[0] = new_radius * np.cos(body_frame_angle)
        foot[1] = new_radius * np.sin(body_frame_angle)
        
        # Enforce ground contact while respecting workspace limits
        # Target: foot_body_z = -base_z for ground contact
        # Constraint: foot_body_z >= min_allowable_body_z to prevent over-extension
        target_body_z = -self.root_pos[2]
        foot[2] = max(target_body_z, self.min_allowable_body_z)
        
        return foot