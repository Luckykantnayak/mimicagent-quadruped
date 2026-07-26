from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_PINWHEEL_LATERAL_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Pinwheel lateral spin gait with synchronized leg rotation.

    - All four legs rotate clockwise (viewed from above) in synchronized pinwheel pattern
    - Each leg sweeps through 360 degrees around its hip attachment point per cycle
    - Base translates leftward continuously with zero yaw rotation
    - All feet maintain ground contact throughout (stance phase only)
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for smooth circular motion
        
        # Base foot positions (hip-relative reference points)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Store hip positions (assumed to be at base of each leg)
        self.hip_positions = {}
        for leg_name in self.leg_names:
            # Hip position is inferred from initial foot position lateral/longitudinal offset
            # Assuming hips are above the feet at z=0 in body frame
            foot = self.base_feet_pos_body[leg_name]
            self.hip_positions[leg_name] = np.array([foot[0], foot[1], 0.0])
        
        # Pinwheel rotation parameters
        self.rotation_radius = 0.25  # Distance from hip to foot during rotation
        self.ground_height = -0.3  # Nominal ground contact height in body frame
        
        # All legs synchronized - same phase offset (no offset)
        self.phase_offsets = {
            leg_names[0]: 0.0,
            leg_names[1]: 0.0,
            leg_names[2]: 0.0,
            leg_names[3]: 0.0,
        }
        
        # Initial angle for each leg based on its hip position
        # Calculate initial angle from hip to foot
        self.initial_angles = {}
        for leg_name in self.leg_names:
            hip = self.hip_positions[leg_name]
            foot = self.base_feet_pos_body[leg_name]
            dx = foot[0] - hip[0]
            dy = foot[1] - hip[1]
            self.initial_angles[leg_name] = np.arctan2(dy, dx)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Lateral velocity parameters (leftward = negative y in body frame)
        self.lateral_velocity = -0.15  # Constant leftward velocity

    def update_base_motion(self, phase, dt):
        """
        Update base with constant leftward lateral velocity and zero rotation.
        """
        # Constant leftward velocity (negative y-axis in body frame)
        self.vel_world = np.array([0.0, self.lateral_velocity, 0.0])
        
        # No angular velocity - maintain forward orientation
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        # Integrate pose in world frame
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position as it rotates in a circular pinwheel pattern.
        
        Each leg rotates clockwise (viewed from above) around its hip attachment point.
        Phase 0.0 -> 1.0 corresponds to 0 -> 360 degrees rotation.
        """
        # Get leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Convert phase to rotation angle (0 to 2π)
        # Clockwise rotation when viewed from above means negative angle progression in standard convention
        rotation_angle = self.initial_angles[leg_name] + 2 * np.pi * leg_phase
        
        # Get hip position
        hip = self.hip_positions[leg_name]
        
        # Calculate foot position in horizontal plane rotating around hip
        foot_x = hip[0] + self.rotation_radius * np.cos(rotation_angle)
        foot_y = hip[1] + self.rotation_radius * np.sin(rotation_angle)
        
        # Maintain ground contact - adjust vertical position
        # As leg rotates, we need to account for changing leg geometry
        # Use a simple model: vertical position adjusts to maintain realistic leg extension
        # Ground height maintained with slight variation based on radial distance
        foot_z = self.ground_height
        
        # Small vertical modulation to account for leg kinematic constraints
        # Legs slightly raise/lower as they sweep to different angular positions
        vertical_modulation = 0.02 * np.sin(2 * np.pi * leg_phase)
        foot_z += vertical_modulation
        
        foot_position = np.array([foot_x, foot_y, foot_z])
        
        return foot_position