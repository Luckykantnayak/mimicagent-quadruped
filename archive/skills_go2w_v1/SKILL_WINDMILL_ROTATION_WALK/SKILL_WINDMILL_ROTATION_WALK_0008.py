from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_WINDMILL_ROTATION_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Windmill rotation walk gait.
    
    - Each leg traces a full vertical circle (windmill blade) in the body frame sagittal plane
    - Right side legs (FR, RR) rotate in sync
    - Left side legs (FL, RL) rotate in sync with 180-degree phase offset from right side
    - Lower semicircle of rotation provides stance/thrust, upper semicircle is swing
    - Base moves forward with constant velocity matching the rearward sweep of stance legs
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Windmill rotation frequency (Hz)
        
        # Windmill circle parameters (body frame)
        # Iteration 2 and 3 achieved best results (reward -2.0, only airtime violation) with offset 0.08-0.12
        # Lowering slightly from 0.08 to 0.06 to achieve ground contact without excessive penetration
        self.circle_radius = 0.09  # Validated to avoid joint limits
        self.circle_center_x_offset = -0.02  # Slight rearward offset for balanced reach
        self.circle_center_z_offset = 0.06  # Compromise between 0.08 (no contact) and 0.01 (too much penetration)
        
        # Base foot positions (nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compute circle centers for each leg
        self.circle_centers = {}
        for leg in self.leg_names:
            base_pos = self.base_feet_pos_body[leg]
            self.circle_centers[leg] = np.array([
                base_pos[0] + self.circle_center_x_offset,
                base_pos[1],  # No lateral offset - windmill rotates in sagittal plane
                base_pos[2] + self.circle_center_z_offset
            ])
        
        # Phase offsets: right legs at 0.0, left legs at 0.5 (restore proven structure)
        self.phase_offsets = {}
        for leg in self.leg_names:
            if leg in ['FR', 'RR']:
                self.phase_offsets[leg] = 0.0  # Right side
            else:  # FL or RL
                self.phase_offsets[leg] = 0.5  # Left side (180 degrees out of phase)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Forward velocity tuned for contact maintenance
        self.forward_velocity = 0.05  # Conservative speed to allow proper ground contact

    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward velocity and level orientation.
        """
        vx = self.forward_velocity
        vy = 0.0
        vz = 0.0
        
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position tracing a vertical circle (windmill blade).
        
        Restore proven parameterization from iterations 2-3:
        - theta = 2*pi*leg_phase + pi/2 offset
        - Right legs (phase_offset 0.0) and left legs (phase_offset 0.5) alternate
        - Circle parameters tuned to achieve ground contact at lowest point
        """
        
        # Get leg-specific phase with offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Circle center for this leg
        center = self.circle_centers[leg_name]
        
        # Convert phase to angle - restore proven pi/2 offset
        theta = 2.0 * np.pi * leg_phase + np.pi / 2.0
        
        # Parametric circle in sagittal plane (x-z)
        x_offset = self.circle_radius * np.cos(theta)
        z_offset = self.circle_radius * np.sin(theta)
        
        # Foot position in body frame
        foot_pos = np.array([
            center[0] + x_offset,
            center[1],  # No lateral motion - windmill rotates in place
            center[2] + z_offset
        ])
        
        return foot_pos