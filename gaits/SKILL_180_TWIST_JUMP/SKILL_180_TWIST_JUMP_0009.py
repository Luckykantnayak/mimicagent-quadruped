from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_180_TWIST_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    180-degree twist jump: compress, explosive jump with yaw rotation,
    tuck legs mid-air, extend for landing facing opposite direction.
    
    Phase breakdown:
      0.00-0.18: Compression (crouch down, initiate yaw)
      0.18-0.38: Launch (explosive extension, lift off)
      0.38-0.68: Aerial rotation (tucked legs, sustained yaw)
      0.68-0.92: Pre-landing (extend legs gradually, reduce yaw rate)
      0.92-1.00: Landing absorption (contact, stabilize, return to neutral)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0

        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Motion parameters - tuned for joint limits, ground clearance, and phase continuity
        self.compression_depth = 0.09  # Moderate compression during crouch (m)
        self.tuck_height = 0.09  # Moderate tuck during aerial phase (m)
        self.tuck_inward = 0.035  # Minimal inward tuck (m)
        self.landing_extension = 0.065  # Increased extension for better ground clearance (m)

        # Base velocity parameters - tuned to stay within height envelope
        self.launch_vz = 1.24  # Upward velocity during launch (m/s)
        self.peak_yaw_rate = 6.6  # Peak yaw rate for 180° rotation (rad/s)

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.accumulated_yaw = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base motion: vertical velocity and yaw rate based on phase.
        Tuned for height envelope compliance and smooth landing.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0

        # Phase 0.0-0.18: Compression
        if phase < 0.18:
            local_phase = phase / 0.18
            # Smooth downward motion during compression
            vz = -0.32 * np.sin(np.pi * local_phase)
            # Smooth yaw rate ramp
            yaw_rate = self.peak_yaw_rate * 0.12 * (local_phase ** 2)

        # Phase 0.18-0.38: Launch
        elif phase < 0.38:
            local_phase = (phase - 0.18) / 0.2
            # Strong upward velocity with smooth decay
            vz = self.launch_vz * (1.0 - 0.18 * local_phase)
            # Establish high yaw rate with smooth transition
            yaw_rate = self.peak_yaw_rate * (0.12 + 0.88 * (1.0 - (1.0 - local_phase) ** 2))

        # Phase 0.38-0.68: Aerial rotation
        elif phase < 0.68:
            local_phase = (phase - 0.38) / 0.3
            # Ballistic trajectory with reduced peak velocity
            vz = self.launch_vz * (0.72 - 2.35 * local_phase)
            # Sustained high yaw rate
            yaw_rate = self.peak_yaw_rate * (1.0 + 0.08 * np.sin(np.pi * local_phase))

        # Phase 0.68-0.92: Pre-landing (extended for gradual descent)
        elif phase < 0.92:
            local_phase = (phase - 0.68) / 0.24
            # Controlled downward velocity with smooth deceleration
            vz = -self.launch_vz * 0.32 * (0.35 + 0.65 * local_phase * (1.0 - 0.35 * local_phase))
            # Yaw rate decreasing smoothly
            yaw_rate = self.peak_yaw_rate * ((1.0 - local_phase) ** 2.2)

        # Phase 0.92-1.0: Landing absorption (gentler descent)
        else:
            local_phase = (phase - 0.92) / 0.08
            # Reduced velocity damping coefficient for gentler ground approach
            decay_factor = (1.0 - local_phase) ** 3.5
            vz = -self.launch_vz * 0.08 * decay_factor
            # Rapid yaw damping
            yaw_rate = self.peak_yaw_rate * 0.02 * (1.0 - local_phase) ** 3.5

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

        self.accumulated_yaw += yaw_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        All legs move symmetrically with smooth trajectories.
        Landing phase maintains extension longer before returning to neutral.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()

        # Determine inward direction (toward body center)
        inward_x_factor = -np.sign(base_pos[0]) if abs(base_pos[0]) > 0.01 else 0.0
        inward_y_factor = -np.sign(base_pos[1]) if abs(base_pos[1]) > 0.01 else 0.0

        # Phase 0.0-0.18: Compression
        if phase < 0.18:
            local_phase = phase / 0.18
            # Smooth compression using cosine
            compression_progress = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            # Retract upward and slightly inward
            foot[2] += self.compression_depth * compression_progress
            foot[0] += inward_x_factor * 0.018 * compression_progress
            foot[1] += inward_y_factor * 0.018 * compression_progress

        # Phase 0.18-0.38: Launch
        elif phase < 0.38:
            local_phase = (phase - 0.18) / 0.2
            # Smooth extension from compressed state
            extension_progress = 0.5 * (1.0 + np.cos(np.pi * local_phase))
            compression_offset = self.compression_depth * extension_progress
            foot[2] += compression_offset
            foot[0] += inward_x_factor * 0.018 * extension_progress
            foot[1] += inward_y_factor * 0.018 * extension_progress

        # Phase 0.38-0.68: Aerial rotation (tucked)
        elif phase < 0.68:
            local_phase = (phase - 0.38) / 0.3
            # Smooth tuck with gradual entry
            tuck_progress = 0.5 * (1.0 - np.cos(np.pi * min(local_phase / 0.6, 1.0)))
            foot[2] += self.tuck_height * tuck_progress
            foot[0] += inward_x_factor * self.tuck_inward * tuck_progress
            foot[1] += inward_y_factor * self.tuck_inward * tuck_progress

        # Phase 0.68-0.92: Pre-landing (blended untuck and extend)
        elif phase < 0.92:
            local_phase = (phase - 0.68) / 0.24
            
            # Smooth untuck progression using cosine
            untuck_progress = 0.5 * (1.0 + np.cos(np.pi * (1.0 - local_phase)))
            
            # Smooth extension progression with gradual acceleration
            extension_progress = local_phase ** 1.4
            
            # Apply untuck (reducing tuck offsets)
            foot[2] += self.tuck_height * untuck_progress
            foot[0] += inward_x_factor * self.tuck_inward * untuck_progress
            foot[1] += inward_y_factor * self.tuck_inward * untuck_progress
            
            # Apply landing extension (moving downward from base position)
            foot[2] -= self.landing_extension * extension_progress

        # Phase 0.92-1.0: Landing absorption with delayed return to neutral
        else:
            local_phase = (phase - 0.92) / 0.08
            
            # Maintain full extension through first 30% of landing phase
            # Then gradually transition to neutral position for phase continuity
            if local_phase < 0.3:
                # Hold full extension during initial ground contact
                extension_remaining = self.landing_extension
            else:
                # Gradually reduce extension to zero by phase 1.0
                retract_progress = (local_phase - 0.3) / 0.7
                extension_remaining = self.landing_extension * (1.0 - retract_progress)
            
            foot[2] -= extension_remaining

        return foot