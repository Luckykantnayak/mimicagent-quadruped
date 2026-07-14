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
        # Key insight: original iteration had ground contact (648 penetrations) with offset=-0.05, radius=0.15
        # That configuration had lowest_point = base_z - 0.05 - 0.15 = base_z - 0.20
        # Reducing radius to 0.09 prevents joint limits
        # Need offset such that lowest_point = base_z + offset - radius is slightly below ground
        # If base_z is at/near ground (z~0), need: offset - radius ≈ -0.01
        # Therefore: offset ≈ radius - 0.01 = 0.09 - 0.01 = 0.08 was tried but failed
        # This suggests base_z is NOT at ground level, likely negative
        # Try offset closer to zero or negative to achieve ground contact
        self.circle_radius = 0.09  # Validated to avoid joint limits
        self.circle_center_x_offset = -0.02  # Slight rearward offset for balanced reach
        self.circle_center_z_offset = 0.01  # Much lower than 0.08, closer to nominal foot height
        
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
        
        # Phase offsets: right legs at 0.25, left legs at 0.75 for better overlap
        # This shifts timing so both groups spend more time near ground
        self.phase_offsets = {}
        for leg in self.leg_names:
            if leg in ['FR', 'RR']:
                self.phase_offsets[leg] = 0.25  # Right side
            else:  # FL or RL
                self.phase_offsets[leg] = 0.75  # Left side (180 degrees out of phase)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Forward velocity reduced to ensure feet can maintain contact
        # Original calculation: radius * 2π * freq * 0.5 ≈ 0.14 m/s was too fast
        # Reduce significantly to allow proper ground contact
        self.forward_velocity = 0.06  # Conservative forward speed

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
        
        Revised parameterization to ensure ground contact:
        - theta maps phase to circular motion with offset adjusted for contact
        - The goal is to have feet at lowest point during rearward portion of swing
        - Using standard circular motion: x = r*cos(theta), z = r*sin(theta)
        - theta = 2*pi*leg_phase creates full rotation per cycle
        
        With phase_offset = 0.25 for right, 0.75 for left:
        - Right legs at global phase=0: leg_phase=0.25, theta=pi/2 (top-center)
        - Left legs at global phase=0: leg_phase=0.75, theta=3pi/2 (bottom-center)
        - At global phase=0.5: right at 0.75->3pi/2 (bottom), left at 0.25->pi/2 (top)
        - This creates alternating vertical positioning
        """
        
        # Get leg-specific phase with offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Circle center for this leg
        center = self.circle_centers[leg_name]
        
        # Convert phase to angle (radians)
        # Remove the pi/2 offset to align bottom with different phase position
        theta = 2.0 * np.pi * leg_phase
        
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