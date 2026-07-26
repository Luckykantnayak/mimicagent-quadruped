from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_180_TWIST_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    180-degree twist jump: compress, explosive jump with yaw rotation,
    tuck legs mid-air, extend for landing facing opposite direction.
    
    Phase breakdown:
      0.00-0.15: Compression (crouch down, initiate yaw)
      0.15-0.35: Launch (explosive extension, lift off)
      0.35-0.65: Aerial rotation (tucked legs, sustained yaw)
      0.65-0.85: Pre-landing (extend legs, reduce yaw rate)
      0.85-1.00: Landing absorption (contact, stabilize)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0  # Full maneuver cycle at 1 Hz

        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Motion parameters - REDUCED to prevent base height violation
        self.compression_depth = 0.12  # Increased upward retraction during crouch (m)
        self.tuck_height = 0.12  # Reduced tuck during aerial phase (m)
        self.tuck_inward = 0.05  # Reduced inward tuck (m)
        self.landing_extension = 0.12  # Downward extension for landing (m)

        # Base velocity parameters - REDUCED to stay within height envelope
        self.launch_vz = 1.3  # Reduced upward velocity during launch (m/s)
        self.peak_yaw_rate = 6.5  # Slightly increased to maintain rotation with less airtime
        self.total_yaw_target = np.pi  # 180 degrees in radians

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Track accumulated yaw for debugging/verification
        self.accumulated_yaw = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base motion: vertical velocity and yaw rate based on phase.
        
        Compression: vz slightly negative, yaw rate ramps up
        Launch: vz positive (explosive), yaw rate high
        Aerial: vz ballistic (up then down), yaw rate sustained
        Pre-landing: vz downward, yaw rate decreasing
        Landing: vz damped to zero quickly, yaw rate to zero
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0

        # Phase 0.0-0.15: Compression
        if phase < 0.15:
            local_phase = phase / 0.15
            # Downward motion during compression using smooth profile
            vz = -0.4 * np.sin(np.pi * local_phase)
            # Ramp up yaw rate smoothly
            yaw_rate = self.peak_yaw_rate * 0.2 * (local_phase ** 2)

        # Phase 0.15-0.35: Launch
        elif phase < 0.35:
            local_phase = (phase - 0.15) / 0.2
            # Strong upward velocity with smooth decay
            vz = self.launch_vz * (1.0 - 0.25 * local_phase)
            # High yaw rate established with smooth ramp
            yaw_rate = self.peak_yaw_rate * (0.2 + 0.8 * (1.0 - (1.0 - local_phase) ** 2))

        # Phase 0.35-0.65: Aerial rotation
        elif phase < 0.65:
            local_phase = (phase - 0.35) / 0.3
            # Ballistic trajectory: controlled arc
            vz = self.launch_vz * (0.75 - 2.3 * local_phase)
            # Sustained high yaw rate with smooth peak
            yaw_rate = self.peak_yaw_rate * (1.0 + 0.15 * np.sin(np.pi * local_phase))

        # Phase 0.65-0.85: Pre-landing
        elif phase < 0.85:
            local_phase = (phase - 0.65) / 0.2
            # Downward velocity controlled
            vz = -self.launch_vz * 0.45 * (0.3 + 0.7 * local_phase)
            # Yaw rate decreasing smoothly
            yaw_rate = self.peak_yaw_rate * (1.0 - local_phase) ** 2

        # Phase 0.85-1.0: Landing absorption - EARLY DAMPING
        else:
            local_phase = (phase - 0.85) / 0.15
            # Rapidly damp downward velocity to zero early in landing phase
            decay_factor = (1.0 - local_phase) ** 3
            vz = -self.launch_vz * 0.3 * decay_factor
            # Rapidly damp yaw rate to zero
            yaw_rate = self.peak_yaw_rate * 0.05 * (1.0 - local_phase) ** 2

        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])

        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

        # Track accumulated yaw for verification
        self.accumulated_yaw += yaw_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        All legs move symmetrically: compress, launch, tuck, extend DOWN for landing.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()

        # Determine inward direction (toward body center)
        inward_x_factor = -np.sign(base_pos[0]) if abs(base_pos[0]) > 0.01 else 0.0
        inward_y_factor = -np.sign(base_pos[1]) if abs(base_pos[1]) > 0.01 else 0.0

        # Phase 0.0-0.15: Compression
        if phase < 0.15:
            local_phase = phase / 0.15
            # Smooth compression profile
            compression_progress = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            # Retract upward and slightly inward
            foot[2] += self.compression_depth * compression_progress
            foot[0] += inward_x_factor * 0.025 * compression_progress
            foot[1] += inward_y_factor * 0.025 * compression_progress

        # Phase 0.15-0.35: Launch
        elif phase < 0.35:
            local_phase = (phase - 0.15) / 0.2
            # Smooth extension from compressed state using cosine
            extension_progress = 0.5 * (1.0 + np.cos(np.pi * local_phase))
            compression_offset = self.compression_depth * extension_progress
            foot[2] += compression_offset
            foot[0] += inward_x_factor * 0.025 * extension_progress
            foot[1] += inward_y_factor * 0.025 * extension_progress

        # Phase 0.35-0.65: Aerial rotation (tucked)
        elif phase < 0.65:
            local_phase = (phase - 0.35) / 0.3
            # Smooth tuck entry and hold
            tuck_progress = 0.5 * (1.0 - np.cos(np.pi * min(local_phase / 0.4, 1.0)))
            foot[2] += self.tuck_height * tuck_progress
            foot[0] += inward_x_factor * self.tuck_inward * tuck_progress
            foot[1] += inward_y_factor * self.tuck_inward * tuck_progress

        # Phase 0.65-0.85: Pre-landing (extend DOWNWARD from base)
        elif phase < 0.85:
            local_phase = (phase - 0.65) / 0.2
            # Transition from tucked to EXTENDED landing position
            # First untuck (0.0 to 0.5 of local_phase)
            # Then extend beyond base (0.5 to 1.0 of local_phase)
            if local_phase < 0.5:
                # Untuck phase
                untuck_progress = 1.0 - 2.0 * local_phase
                foot[2] += self.tuck_height * untuck_progress
                foot[0] += inward_x_factor * self.tuck_inward * untuck_progress
                foot[1] += inward_y_factor * self.tuck_inward * untuck_progress
            else:
                # Extension phase - move feet DOWNWARD
                extend_progress = (local_phase - 0.5) / 0.5
                foot[2] -= self.landing_extension * extend_progress

        # Phase 0.85-1.0: Landing absorption (feet EXTENDED DOWNWARD)
        else:
            local_phase = (phase - 0.85) / 0.15
            # Maintain extended position then compress slightly for absorption
            # Start fully extended, compress upward slightly as weight settles
            absorption_progress = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            # Full extension at start, partial compression at end
            current_extension = self.landing_extension * (1.0 - 0.3 * absorption_progress)
            foot[2] -= current_extension

        return foot