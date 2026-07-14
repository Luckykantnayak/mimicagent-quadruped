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

        # Motion parameters
        self.compression_depth = 0.08  # Upward retraction during crouch (m)
        self.tuck_height = 0.15  # Upward tuck during aerial phase (m)
        self.tuck_inward = 0.06  # Inward tuck to reduce moment of inertia (m)
        self.landing_compression = 0.05  # Slight compression on landing (m)

        # Base velocity parameters
        self.launch_vz = 1.8  # Upward velocity during launch (m/s)
        self.peak_yaw_rate = 6.3  # Peak yaw rate to achieve ~180° rotation (rad/s)
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
        Landing: vz damped to zero, yaw rate to zero
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0

        # Phase 0.0-0.15: Compression
        if phase < 0.15:
            local_phase = phase / 0.15
            # Slight downward motion during compression
            vz = -0.3 * np.sin(np.pi * local_phase)
            # Ramp up yaw rate
            yaw_rate = self.peak_yaw_rate * 0.3 * local_phase

        # Phase 0.15-0.35: Launch
        elif phase < 0.35:
            local_phase = (phase - 0.15) / 0.2
            # Strong upward velocity
            vz = self.launch_vz * (1.0 - 0.3 * local_phase)
            # High yaw rate established
            yaw_rate = self.peak_yaw_rate * (0.3 + 0.7 * local_phase)

        # Phase 0.35-0.65: Aerial rotation
        elif phase < 0.65:
            local_phase = (phase - 0.35) / 0.3
            # Ballistic trajectory: upward velocity decreases, becomes downward
            # Apex around mid-phase (local_phase ~ 0.5)
            vz = self.launch_vz * (0.7 - 2.5 * local_phase)
            # Sustained high yaw rate (peak during mid-flight)
            yaw_rate = self.peak_yaw_rate * (1.0 + 0.2 * np.sin(np.pi * local_phase))

        # Phase 0.65-0.85: Pre-landing
        elif phase < 0.85:
            local_phase = (phase - 0.65) / 0.2
            # Downward velocity increasing
            vz = -self.launch_vz * 0.5 * (0.5 + local_phase)
            # Yaw rate decreasing as legs extend
            yaw_rate = self.peak_yaw_rate * (1.0 - local_phase)

        # Phase 0.85-1.0: Landing absorption
        else:
            local_phase = (phase - 0.85) / 0.15
            # Damp downward velocity to zero
            vz = -self.launch_vz * 0.5 * (1.5 - local_phase) * (1.0 - local_phase)
            # Rapidly damp yaw rate to zero
            yaw_rate = self.peak_yaw_rate * (1.0 - local_phase) * 0.1

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
        All legs move symmetrically: compress, launch, tuck, extend, land.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()

        # Determine inward direction (toward body center)
        # Front legs have positive x, rear legs negative x
        # Left legs have positive y, right legs negative y
        inward_x_factor = -np.sign(base_pos[0]) if abs(base_pos[0]) > 0.01 else 0.0
        inward_y_factor = -np.sign(base_pos[1]) if abs(base_pos[1]) > 0.01 else 0.0

        # Phase 0.0-0.15: Compression
        if phase < 0.15:
            local_phase = phase / 0.15
            # Retract upward and slightly inward
            foot[2] += self.compression_depth * local_phase
            foot[0] += inward_x_factor * 0.02 * local_phase
            foot[1] += inward_y_factor * 0.02 * local_phase

        # Phase 0.15-0.35: Launch
        elif phase < 0.35:
            local_phase = (phase - 0.15) / 0.2
            # Extend downward and outward from compressed state
            compression_offset = self.compression_depth * (1.0 - local_phase)
            foot[2] += compression_offset
            foot[0] += inward_x_factor * 0.02 * (1.0 - local_phase)
            foot[1] += inward_y_factor * 0.02 * (1.0 - local_phase)

        # Phase 0.35-0.65: Aerial rotation (tucked)
        elif phase < 0.65:
            local_phase = (phase - 0.35) / 0.3
            # Tuck inward and upward to reduce rotational inertia
            tuck_progress = np.sin(np.pi * min(local_phase / 0.3, 1.0))  # Smooth entry
            foot[2] += self.tuck_height * tuck_progress
            foot[0] += inward_x_factor * self.tuck_inward * tuck_progress
            foot[1] += inward_y_factor * self.tuck_inward * tuck_progress

        # Phase 0.65-0.85: Pre-landing (extend)
        elif phase < 0.85:
            local_phase = (phase - 0.65) / 0.2
            # Extend from tucked to landing position
            tuck_progress = 1.0 - local_phase
            foot[2] += self.tuck_height * tuck_progress
            foot[0] += inward_x_factor * self.tuck_inward * tuck_progress
            foot[1] += inward_y_factor * self.tuck_inward * tuck_progress

        # Phase 0.85-1.0: Landing absorption
        else:
            local_phase = (phase - 0.85) / 0.15
            # Slight compression to absorb landing impact
            compression = self.landing_compression * np.sin(np.pi * local_phase * 0.5)
            foot[2] += compression

        return foot