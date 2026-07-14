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
        # Circle center is offset behind and below the nominal foot position
        self.circle_radius = 0.15  # Radius of windmill circle
        self.circle_center_x_offset = -0.05  # Center offset forward from base foot position
        self.circle_center_z_offset = -0.05  # Center offset down from base foot position
        
        # Base foot positions (nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compute circle centers for each leg
        self.circle_centers = {}
        for leg in self.leg_names:
            base_pos = self.base_feet_pos_body[leg]
            self.circle_centers[leg] = np.array([
                base_pos[0] + self.circle_center_x_offset,
                base_pos[1],  # No lateral offset
                base_pos[2] + self.circle_center_z_offset
            ])
        
        # Phase offsets: right legs (FR, RR) at 0.0, left legs (FL, RL) at 0.5
        self.phase_offsets = {}
        for leg in self.leg_names:
            if leg.startswith('FR') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0  # Right side
            else:  # FL or RL
                self.phase_offsets[leg] = 0.5  # Left side (180 degrees offset)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Forward velocity tuned to match rearward sweep speed during stance
        # Average rearward velocity during stance = circle_radius * 2*pi*freq / 2
        self.forward_velocity = self.circle_radius * 2 * np.pi * self.freq * 0.5

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
        
        Circle parameterization:
        - phase = 0.0: rear-bottom (start of stance for left legs)
        - phase = 0.25: top-rear (peak of swing)
        - phase = 0.5: front-bottom (start of stance for right legs)
        - phase = 0.75: bottom-front (mid stance)
        
        Circle is traced clockwise when viewed from right side:
        - Lower semicircle (phase offset ~0.5-1.0 for left, 0.0-0.5 for right): stance
        - Upper semicircle: swing
        
        Parametric equations:
        x(theta) = center_x + radius * cos(theta)
        z(theta) = center_z + radius * sin(theta)
        where theta = 2*pi*leg_phase, starting from right (0 rad)
        
        To get desired phasing (rear-bottom at phase=0 for left legs):
        theta = 2*pi*leg_phase - pi/2 (rotates circle so phase=0 starts at bottom-rear)
        """
        
        # Get leg-specific phase with offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Circle center for this leg
        center = self.circle_centers[leg_name]
        
        # Convert phase to angle (radians)
        # Offset by -pi/2 so phase=0 corresponds to rear-bottom position
        # Additional pi offset to make rotation direction match forward walk
        theta = 2 * np.pi * leg_phase + np.pi / 2
        
        # Parametric circle in sagittal plane (x-z)
        x_offset = self.circle_radius * np.cos(theta)
        z_offset = self.circle_radius * np.sin(theta)
        
        # Foot position in body frame
        foot_pos = np.array([
            center[0] + x_offset,
            center[1],  # No lateral motion
            center[2] + z_offset
        ])
        
        return foot_pos