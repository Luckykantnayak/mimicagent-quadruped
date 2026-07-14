from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_REVERSE_HELIX_SINK_MotionGenerator(BaseMotionGenerator):
    """
    Reverse helix sink motion: robot moves backward while rotating counter-clockwise
    and gradually lowering its base height, tracing a descending helical path.
    
    - Base moves backward (negative vx) with sustained velocity
    - Base rotates counter-clockwise (positive yaw rate) completing 360° per cycle
    - Base descends (negative vz) progressively to minimum height
    - All four feet maintain ground contact throughout
    - Legs compress and spread to accommodate the multi-axis motion
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.4  # Controlled frequency for smooth helical motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Store initial world-frame foot positions (assume ground at z=0)
        self.initial_feet_world = {}
        for leg in leg_names:
            foot_body = self.base_feet_pos_body[leg]
            foot_world = quat_rotate(self.root_quat, foot_body) + self.root_pos
            foot_world[2] = 0.0  # Enforce ground contact
            self.initial_feet_world[leg] = foot_world.copy()
        
        # Motion parameters - reduced for gentler descent
        # Backward velocity profile
        self.vx_phase1 = -0.25
        self.vx_phase2 = -0.35
        self.vx_phase3 = -0.30
        self.vx_phase4 = -0.25
        
        # Yaw rotation parameters (360 degrees per cycle)
        self.yaw_rate_phase1 = 1.8
        self.yaw_rate_phase2 = 2.0
        self.yaw_rate_phase3 = 1.9
        self.yaw_rate_phase4 = 1.7
        
        # Vertical descent parameters - significantly reduced to prevent ground penetration
        self.vz_phase1 = -0.08
        self.vz_phase2 = -0.12
        self.vz_phase3 = -0.06
        self.vz_phase4 = -0.02
        
        # Leg parameters
        self.stance_width_expansion = 0.04  # Moderate lateral spreading
        self.foot_height_offset = 0.01  # Small clearance above ground

    def update_base_motion(self, phase, dt):
        """
        Update base with backward translation, counter-clockwise yaw, and vertical descent.
        """
        # Smooth phase-based velocity interpolation
        if phase < 0.25:
            sub_phase = smooth_step(phase / 0.25)
            vx = self.vx_phase1
            yaw_rate = self.yaw_rate_phase1
            vz = self.vz_phase1
        elif phase < 0.5:
            sub_phase = smooth_step((phase - 0.25) / 0.25)
            vx = self.vx_phase1 + (self.vx_phase2 - self.vx_phase1) * sub_phase
            yaw_rate = self.yaw_rate_phase1 + (self.yaw_rate_phase2 - self.yaw_rate_phase1) * sub_phase
            vz = self.vz_phase1 + (self.vz_phase2 - self.vz_phase1) * sub_phase
        elif phase < 0.75:
            sub_phase = smooth_step((phase - 0.5) / 0.25)
            vx = self.vx_phase2 + (self.vx_phase3 - self.vx_phase2) * sub_phase
            yaw_rate = self.yaw_rate_phase2 + (self.yaw_rate_phase3 - self.yaw_rate_phase2) * sub_phase
            vz = self.vz_phase2 + (self.vz_phase3 - self.vz_phase2) * sub_phase
        else:
            sub_phase = smooth_step((phase - 0.75) / 0.25)
            vx = self.vx_phase3 + (self.vx_phase4 - self.vx_phase3) * sub_phase
            yaw_rate = self.yaw_rate_phase3 + (self.yaw_rate_phase4 - self.yaw_rate_phase3) * sub_phase
            vz = self.vz_phase3 + (self.vz_phase4 - self.vz_phase3) * sub_phase
        
        # Set velocities in world frame
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
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
        Compute foot position in body frame maintaining ground contact.
        Feet track world-frame ground plane while body rotates and descends.
        """
        # Get initial world-frame foot position
        initial_world = self.initial_feet_world[leg_name].copy()
        
        # Stance width expansion for stability (in world frame)
        if phase < 0.5:
            expansion_factor = smooth_step(phase / 0.5)
        else:
            expansion_factor = 1.0
        
        lateral_expansion = self.stance_width_expansion * expansion_factor
        
        # Apply lateral expansion based on leg position
        if 'FL' in leg_name:
            initial_world[1] += lateral_expansion
        elif 'FR' in leg_name:
            initial_world[1] -= lateral_expansion
        elif 'RL' in leg_name:
            initial_world[1] += lateral_expansion
        elif 'RR' in leg_name:
            initial_world[1] -= lateral_expansion
        
        # Maintain ground contact with small offset
        target_world = initial_world.copy()
        target_world[2] = self.foot_height_offset
        
        # Transform target from world frame to current body frame
        # foot_world = quat_rotate(root_quat, foot_body) + root_pos
        # Inverse: foot_body = quat_rotate(quat_inverse(root_quat), foot_world - root_pos)
        
        root_quat_inv = quat_inverse(self.root_quat)
        foot_body = quat_rotate(root_quat_inv, target_world - self.root_pos)
        
        return foot_body


def smooth_step(x):
    """Smooth interpolation function (3x^2 - 2x^3) for continuous derivatives."""
    x = np.clip(x, 0.0, 1.0)
    return 3.0 * x * x - 2.0 * x * x * x


def quat_inverse(q):
    """Compute quaternion inverse (conjugate for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])