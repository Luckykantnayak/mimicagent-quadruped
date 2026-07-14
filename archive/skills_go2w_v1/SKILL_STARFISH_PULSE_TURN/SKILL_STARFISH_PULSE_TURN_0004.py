from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_STARFISH_PULSE_TURN_MotionGenerator(BaseMotionGenerator):
    """
    Starfish Pulse Turn: In-place rotational maneuver with synchronized radial leg pulsing.
    
    All four legs simultaneously extend and retract radially while the base executes
    clockwise yaw rotation. Creates a starfish-like pulsing appearance with continuous
    four-point ground contact.
    
    Phase breakdown:
      [0.0, 0.3]: Radial extension + yaw initiation
      [0.3, 0.5]: Extended hold + peak yaw rate
      [0.5, 0.8]: Radial retraction + yaw momentum
      [0.8, 1.0]: Contracted reset + yaw deceleration
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5
        
        # Store nominal foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compute nominal leg lengths from hip to foot
        self.nominal_leg_lengths = {}
        for leg_name, foot_pos in self.base_feet_pos_body.items():
            self.nominal_leg_lengths[leg_name] = np.linalg.norm(foot_pos)
        
        # Radial pulsing parameters
        self.radial_extension_max = 0.12
        self.radial_contraction_max = 0.06
        
        # Yaw rotation parameters
        self.yaw_rate_max = 2.0
        
        # Base height modulation parameters
        self.base_height_amplitude = 0.02
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (world frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def compute_radial_extension_factor(self, phase):
        """
        Compute radial extension factor based on phase.
        Returns:
          > 0.0 during extension phase (legs move outward)
          < 0.0 during retraction phase (legs move inward)
          = 0.0 at nominal stance
        """
        if phase < 0.3:
            # Extension phase: 0.0 -> 0.3
            progress = phase / 0.3
            extension = self.radial_extension_max * (1.0 - np.cos(progress * np.pi)) / 2.0
        elif phase < 0.5:
            # Hold at maximum extension: 0.3 -> 0.5
            extension = self.radial_extension_max
        elif phase < 0.8:
            # Retraction phase: 0.5 -> 0.8
            progress = (phase - 0.5) / 0.3
            extension = self.radial_extension_max * np.cos(progress * np.pi / 2.0) - self.radial_contraction_max * np.sin(progress * np.pi / 2.0)
        else:
            # Contracted phase: 0.8 -> 1.0
            progress = (phase - 0.8) / 0.2
            extension = -self.radial_contraction_max * (1.0 - progress)
        
        return extension

    def compute_yaw_rate(self, phase):
        """
        Compute yaw angular velocity based on phase.
        Smooth ramp up, sustained peak, smooth ramp down.
        """
        if phase < 0.3:
            # Ramp up: 0.0 -> 0.3
            progress = phase / 0.3
            yaw_rate = self.yaw_rate_max * np.sin(progress * np.pi / 2.0)
        elif phase < 0.5:
            # Peak yaw rate: 0.3 -> 0.5
            yaw_rate = self.yaw_rate_max
        elif phase < 0.8:
            # Ramp down: 0.5 -> 0.8
            progress = (phase - 0.5) / 0.3
            yaw_rate = self.yaw_rate_max * np.cos(progress * np.pi / 2.0)
        else:
            # Minimal yaw: 0.8 -> 1.0
            yaw_rate = 0.0
        
        return yaw_rate

    def compute_base_height_offset(self, phase):
        """
        Compute vertical base position offset.
        Base rises during contracted phase to maintain ground contact as legs retract.
        """
        if phase < 0.8:
            height_offset = 0.0
        else:
            # Slight rise during contracted phase: 0.8 -> 1.0
            progress = (phase - 0.8) / 0.2
            height_offset = self.base_height_amplitude * np.sin(progress * np.pi / 2.0)
        
        return height_offset

    def update_base_motion(self, phase, dt):
        """
        Update base pose with in-place rotation and subtle height modulation.
        """
        # Compute height offset derivative for smooth vertical velocity
        phase_prev = phase - self.freq * dt
        if phase_prev < 0.0:
            phase_prev += 1.0
        
        height_offset = self.compute_base_height_offset(phase)
        height_offset_prev = self.compute_base_height_offset(phase_prev)
        
        # Vertical velocity from height offset change
        vz = (height_offset - height_offset_prev) / dt if dt > 0 else 0.0
        self.vel_world = np.array([0.0, 0.0, vz])
        
        # Angular velocity: pure yaw rotation
        yaw_rate = self.compute_yaw_rate(phase)
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
        Compute foot position in body frame with radial pulsing and tangential arc motion.
        
        All legs execute synchronized radial extension/retraction while feet trace
        clockwise circular arcs to maintain contact during yaw rotation.
        Feet Z-coordinate adjusted to maintain ground contact accounting for base height and radial motion.
        """
        # Get nominal foot position in body frame
        nominal_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial direction from body center to foot (in XY plane)
        radial_xy = nominal_pos[:2].copy()
        radial_distance = np.linalg.norm(radial_xy)
        
        if radial_distance > 1e-6:
            radial_dir = radial_xy / radial_distance
        else:
            radial_dir = np.array([1.0, 0.0])
        
        # Compute radial extension
        extension = self.compute_radial_extension_factor(phase)
        
        # Apply radial extension/retraction in XY plane
        foot_pos = nominal_pos.copy()
        foot_pos[0] = nominal_pos[0] + radial_dir[0] * extension
        foot_pos[1] = nominal_pos[1] + radial_dir[1] * extension
        
        # Add tangential circular arc motion to support yaw rotation
        tangential_dir = np.array([-radial_dir[1], radial_dir[0]])
        tangential_amplitude = 0.05
        tangential_offset = tangential_amplitude * np.sin(2.0 * np.pi * phase)
        
        foot_pos[0] += tangential_dir[0] * tangential_offset
        foot_pos[1] += tangential_dir[1] * tangential_offset
        
        # Compute Z position to maintain ground contact
        # Account for base height offset
        base_height_offset = self.compute_base_height_offset(phase)
        
        # Compute current planar distance from hip to foot
        current_planar_distance = np.linalg.norm(foot_pos[:2])
        
        # Compute required Z offset to maintain constant leg length while on ground
        # Using Pythagorean theorem: leg_length^2 = planar_distance^2 + z_offset^2
        nominal_leg_length = self.nominal_leg_lengths[leg_name]
        
        # Ensure we don't exceed leg length in planar distance
        if current_planar_distance < nominal_leg_length:
            # Z component to maintain leg length (negative because foot is below hip)
            z_component = -np.sqrt(nominal_leg_length**2 - current_planar_distance**2)
        else:
            # If planar distance exceeds leg length, use minimal Z
            z_component = nominal_pos[2]
        
        # Adjust for base height changes: when base rises, foot must "lower" in body frame to stay on ground
        foot_pos[2] = z_component - base_height_offset
        
        return foot_pos