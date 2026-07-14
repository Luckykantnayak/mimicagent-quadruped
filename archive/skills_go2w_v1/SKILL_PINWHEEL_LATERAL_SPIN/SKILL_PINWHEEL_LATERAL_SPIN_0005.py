from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_PINWHEEL_LATERAL_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Pinwheel lateral spin gait with synchronized circular leg rotation.

    - All four legs rotate 360° clockwise around their hip joints in body frame
    - Base translates continuously leftward (negative y direction) with zero yaw
    - Diagonal pairs (FL-RR, FR-RL) coordinate contact through phase-dependent radial extension
    - Legs maintain ground contact via adaptive radial modulation during rotation
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for smooth pinwheel rotation
        
        # Base foot positions (hip-relative in body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compute hip positions and rotation radii for each leg
        self.hip_positions = {}
        self.rotation_radii = {}
        self.initial_angles = {}
        
        for leg_name in self.leg_names:
            foot_pos = self.base_feet_pos_body[leg_name]
            # Hip position approximated as (x*0.7, y*0.7, 0) in body frame
            self.hip_positions[leg_name] = np.array([foot_pos[0] * 0.7, foot_pos[1] * 0.7, 0.0])
            
            # Compute initial radius from hip to foot in xy plane
            hip_to_foot = foot_pos - self.hip_positions[leg_name]
            self.rotation_radii[leg_name] = np.sqrt(hip_to_foot[0]**2 + hip_to_foot[1]**2)
            
            # Compute initial angle (for phase=0 alignment)
            self.initial_angles[leg_name] = np.arctan2(hip_to_foot[1], hip_to_foot[0])

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Lateral velocity tuning (negative y = leftward)
        self.vy_lateral = -0.15  # Tuned to match leg rotation lateral component
        
        # Radial extension parameters for contact maintenance
        self.radial_extension_amp = 0.03  # Amplitude of radial modulation
        
        # Diagonal pair phase offset for contact coordination
        # FL-RR are group_1, FR-RL are group_2
        # Phase offset embedded in contact favorability, not absolute phase
        self.contact_phase_offsets = {
            leg_names[0]: 0.0,   # FL
            leg_names[1]: 0.5,   # FR (diagonal offset)
            leg_names[2]: 0.5,   # RL (diagonal offset)
            leg_names[3]: 0.0,   # RR
        }

    def update_base_motion(self, phase, dt):
        """
        Update base with constant leftward velocity and zero angular rates.
        Base heading remains fixed; pinwheel effect from leg rotation only.
        """
        # Constant leftward velocity
        self.vel_world = np.array([0.0, self.vy_lateral, 0.0])
        
        # Zero angular velocity (no base rotation)
        self.omega_world = np.array([0.0, 0.0, 0.0])

        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position via circular rotation around hip in body-frame xy plane.
        
        - Rotation: 360° clockwise over full phase cycle [0,1]
        - Radial extension: modulated to maintain contact during stance-favorable arcs
        - Contact timing: diagonal pairs coordinate via phase-dependent extension
        """
        
        hip_pos = self.hip_positions[leg_name]
        base_radius = self.rotation_radii[leg_name]
        initial_angle = self.initial_angles[leg_name]
        
        # Clockwise rotation: subtract angle (negative direction in standard convention)
        rotation_angle = initial_angle - (2 * np.pi * phase)
        
        # Phase-dependent radial extension for contact maintenance
        # Legs extend more during their stance-favorable arcs (coordinated via diagonal pairing)
        contact_phase = (phase + self.contact_phase_offsets[leg_name]) % 1.0
        
        # Extension profile: maximum at contact_phase = 0.0 and 0.5 (diagonal coordination)
        # Legs extend during forward and lateral arcs, retract during rearward inward arcs
        extension_factor = 1.0 + self.radial_extension_amp * (
            np.cos(4 * np.pi * contact_phase) + 1.0
        ) / 2.0
        
        effective_radius = base_radius * extension_factor
        
        # Compute foot position in body frame via circular trajectory
        foot_x = hip_pos[0] + effective_radius * np.cos(rotation_angle)
        foot_y = hip_pos[1] + effective_radius * np.sin(rotation_angle)
        
        # Vertical position: maintain nominal stance height with slight modulation
        # Z varies slightly to keep contact during rotation (lower during stance-favorable arcs)
        base_z = self.base_feet_pos_body[leg_name][2]
        z_modulation = -0.01 * (np.cos(4 * np.pi * contact_phase) + 1.0) / 2.0
        foot_z = base_z + z_modulation
        
        return np.array([foot_x, foot_y, foot_z])