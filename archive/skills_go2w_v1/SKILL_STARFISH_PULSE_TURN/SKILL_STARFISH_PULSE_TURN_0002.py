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
        self.freq = 0.5  # Slower cycle for dramatic pulsing effect
        
        # Store nominal foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Radial pulsing parameters
        self.radial_extension_max = 0.15  # Maximum outward extension from nominal (meters)
        self.radial_contraction_max = 0.08  # Maximum inward retraction from nominal (meters)
        
        # Yaw rotation parameters
        self.yaw_rate_max = 2.0  # Maximum yaw angular velocity (rad/s)
        
        # Base height modulation parameters (slight rise during contracted phase only)
        self.base_height_amplitude = 0.02  # Subtle vertical rise (meters)
        
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
          > 1.0 during extension phase (legs move outward)
          < 1.0 during retraction phase (legs move inward)
          = 1.0 at nominal stance
        """
        if phase < 0.3:
            # Extension phase: 0.0 -> 0.3
            progress = phase / 0.3
            # Smooth extension using cosine
            extension = self.radial_extension_max * (1.0 - np.cos(progress * np.pi)) / 2.0
        elif phase < 0.5:
            # Hold at maximum extension: 0.3 -> 0.5
            extension = self.radial_extension_max
        elif phase < 0.8:
            # Retraction phase: 0.5 -> 0.8
            progress = (phase - 0.5) / 0.3
            # Smooth retraction using cosine interpolation
            extension = self.radial_extension_max * (1.0 - progress) - self.radial_contraction_max * progress
            extension = self.radial_extension_max * np.cos(progress * np.pi / 2.0) - self.radial_contraction_max * np.sin(progress * np.pi / 2.0)
        else:
            # Contracted phase: 0.8 -> 1.0
            # Smooth transition back to cycle start
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
            # Smooth acceleration using sine curve
            yaw_rate = self.yaw_rate_max * np.sin(progress * np.pi / 2.0)
        elif phase < 0.5:
            # Peak yaw rate: 0.3 -> 0.5
            yaw_rate = self.yaw_rate_max
        elif phase < 0.8:
            # Ramp down: 0.5 -> 0.8
            progress = (phase - 0.5) / 0.3
            # Smooth deceleration
            yaw_rate = self.yaw_rate_max * np.cos(progress * np.pi / 2.0)
        else:
            # Minimal yaw: 0.8 -> 1.0
            progress = (phase - 0.8) / 0.2
            yaw_rate = self.yaw_rate_max * 0.0 * (1.0 - progress)
        
        return yaw_rate

    def compute_base_height_offset(self, phase):
        """
        Compute vertical base position offset.
        Slight rise during contracted phase (0.8 -> 1.0), neutral otherwise.
        """
        if phase < 0.8:
            # Neutral height during extension, hold, and retraction
            height_offset = 0.0
        else:
            # Slight rise during contracted phase: 0.8 -> 1.0
            progress = (phase - 0.8) / 0.2
            # Smooth rise using sine
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
        Foot Z compensates for base height changes to maintain ground contact.
        """
        # Get nominal foot position in body frame
        nominal_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial direction from body center to foot (in XY plane)
        radial_xy = nominal_pos[:2].copy()
        radial_distance = np.linalg.norm(radial_xy)
        
        if radial_distance > 1e-6:
            radial_dir = radial_xy / radial_distance
        else:
            # Fallback for feet at body center
            radial_dir = np.array([1.0, 0.0])
        
        # Compute radial extension
        extension = self.compute_radial_extension_factor(phase)
        
        # Apply radial extension/retraction
        foot_pos = nominal_pos.copy()
        foot_pos[0] = nominal_pos[0] + radial_dir[0] * extension
        foot_pos[1] = nominal_pos[1] + radial_dir[1] * extension
        
        # Add tangential circular arc motion to support yaw rotation
        # Compute tangential direction (perpendicular to radial, clockwise when viewed from above)
        tangential_dir = np.array([-radial_dir[1], radial_dir[0]])
        
        # Tangential displacement proportional to phase
        current_radial = radial_distance + extension
        tangential_amplitude = 0.05  # Tuning parameter for tangential motion
        tangential_offset = tangential_amplitude * np.sin(2.0 * np.pi * phase)
        
        foot_pos[0] += tangential_dir[0] * tangential_offset
        foot_pos[1] += tangential_dir[1] * tangential_offset
        
        # Z position: compensate for base height offset to maintain ground contact
        # When base rises in world frame, foot must lower in body frame to stay on ground
        base_height_offset = self.compute_base_height_offset(phase)
        foot_pos[2] = nominal_pos[2] - base_height_offset
        
        return foot_pos