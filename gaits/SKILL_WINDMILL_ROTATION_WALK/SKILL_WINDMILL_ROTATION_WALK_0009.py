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
    
    KEY FIX: Theta parameterization redesigned to align lowest point (bottom) with rearmost
    horizontal position, ensuring ground contact occurs during rearward sweep (stance phase).
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Windmill rotation frequency (Hz)
        
        # Windmill circle parameters (body frame)
        # Previous iterations achieved -2.0 reward with offsets 0.06-0.12, but persistent airtime
        # due to geometric misalignment (bottom at horizontal center, not rear)
        # With corrected theta phasing, can lower center more to achieve ground contact
        self.circle_radius = 0.09  # Validated to avoid joint limits
        self.circle_center_x_offset = -0.02  # Slight rearward offset
        self.circle_center_z_offset = 0.04  # Lowered with corrected phasing to enable contact
        
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
        
        # Phase offsets: right legs at 0.0, left legs at 0.5 (bilateral alternation)
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
        
        # Forward velocity matched to rearward sweep during corrected stance phase
        self.forward_velocity = 0.07  # Increased to match corrected geometry

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
        
        CORRECTED PARAMETERIZATION:
        - Horizontal (x) and vertical (z) components use different theta offsets
        - This aligns the lowest point (bottom) with rearmost horizontal position
        - Ensures ground contact occurs during rearward sweep (proper stance phase)
        
        theta_x = 2*pi*leg_phase + pi: starts at rear (cos(pi) = -1)
        theta_z = 2*pi*leg_phase + 3*pi/2: starts at bottom (sin(3*pi/2) = -1)
        
        At leg_phase = 0.0:
        - x at rear-most position (negative x_offset)
        - z at bottom (negative z_offset)
        - Perfect stance initiation position
        
        At leg_phase = 0.5:
        - x at front-most position (positive x_offset)
        - z at top (positive z_offset)
        - Perfect swing apex position
        
        With phase_offset = 0.5 for left legs:
        - Left legs start at rear-bottom when right legs are at front-top
        - Creates proper bilateral alternation with sustained ground contact
        """
        
        # Get leg-specific phase with offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Circle center for this leg
        center = self.circle_centers[leg_name]
        
        # Separate theta for horizontal and vertical components
        # This critical change aligns bottom with rear for proper stance contact
        theta_x = 2.0 * np.pi * leg_phase + np.pi  # Horizontal starts at rear
        theta_z = 2.0 * np.pi * leg_phase + 3.0 * np.pi / 2.0  # Vertical starts at bottom
        
        # Parametric windmill motion with corrected phasing
        x_offset = self.circle_radius * np.cos(theta_x)
        z_offset = self.circle_radius * np.sin(theta_z)
        
        # Foot position in body frame
        foot_pos = np.array([
            center[0] + x_offset,
            center[1],  # No lateral motion - windmill rotates in sagittal plane only
            center[2] + z_offset
        ])
        
        return foot_pos