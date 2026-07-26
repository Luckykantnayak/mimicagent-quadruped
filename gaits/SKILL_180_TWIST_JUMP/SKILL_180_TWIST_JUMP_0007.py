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
      0.65-0.90: Pre-landing (extend legs gradually, reduce yaw rate)
      0.90-1.00: Landing absorption (contact, stabilize)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0

        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Motion parameters - tuned for joint limits and ground clearance
        self.compression_depth = 0.10  # Moderate upward retraction during crouch (m)
        self.tuck_height = 0.10  # Moderate tuck during aerial phase (m)
        self.tuck_inward = 0.04  # Moderate inward tuck (m)
        self.landing_extension = 0.07  # Conservative downward extension for landing (m)

        # Base velocity parameters
        self.launch_vz = 1.3  # Upward velocity during launch (m/s)
        self.peak_yaw_rate = 6.5  # Peak yaw rate for 180° rotation (rad/s)

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.accumulated_yaw = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base motion: vertical velocity and yaw rate based on phase.
        Extended pre-landing phase for gentler descent.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0

        # Phase 0.0-0.15: Compression
        if phase < 0.15:
            local_phase = phase / 0.15
            # Smooth downward motion during compression
            vz = -0.35 * np.sin(np.pi * local_phase)
            # Smooth yaw rate ramp
            yaw_rate = self.peak_yaw_rate * 0.15 * (local_phase ** 2)

        # Phase 0.15-0.35: Launch
        elif phase < 0.35:
            local_phase = (phase - 0.15) / 0.2
            # Strong upward velocity with smooth decay
            vz = self.launch_vz * (1.0 - 0.2 * local_phase)
            # Establish high yaw rate
            yaw_rate = self.peak_yaw_rate * (0.15 + 0.85 * (1.0 - (1.0 - local_phase) ** 2))

        # Phase 0.35-0.65: Aerial rotation
        elif phase < 0.65:
            local_phase = (phase - 0.35) / 0.3
            # Ballistic trajectory with controlled arc
            vz = self.launch_vz * (0.8 - 2.4 * local_phase)
            # Sustained high yaw rate
            yaw_rate = self.peak_yaw_rate * (1.0 + 0.1 * np.sin(np.pi * local_phase))

        # Phase 0.65-0.90: Pre-landing (EXTENDED for gradual descent)
        elif phase < 0.90:
            local_phase = (phase - 0.65) / 0.25
            # Controlled downward velocity that decreases approaching landing
            vz = -self.launch_vz * 0.35 * (0.4 + 0.6 * local_phase * (1.0 - 0.3 * local_phase))
            # Yaw rate decreasing smoothly
            yaw_rate = self.peak_yaw_rate * ((1.0 - local_phase) ** 2)

        # Phase 0.90-1.0: Landing absorption (SHORTENED but sufficient)
        else:
            local_phase = (phase - 0.90) / 0.10
            # Rapid velocity damping with cubic decay
            decay_factor = (1.0 - local_phase) ** 3
            vz = -self.launch_vz * 0.15 * decay_factor
            # Rapid yaw damping
            yaw_rate = self.peak_yaw_rate * 0.03 * (1.0 - local_phase) ** 3

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
        Extended pre-landing phase for gradual extension.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()

        # Determine inward direction (toward body center)
        inward_x_factor = -np.sign(base_pos[0]) if abs(base_pos[0]) > 0.01 else 0.0
        inward_y_factor = -np.sign(base_pos[1]) if abs(base_pos[1]) > 0.01 else 0.0

        # Phase 0.0-0.15: Compression
        if phase < 0.15:
            local_phase = phase / 0.15
            # Smooth compression using cosine
            compression_progress = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            # Retract upward and slightly inward
            foot[2] += self.compression_depth * compression_progress
            foot[0] += inward_x_factor * 0.02 * compression_progress
            foot[1] += inward_y_factor * 0.02 * compression_progress

        # Phase 0.15-0.35: Launch
        elif phase < 0.35:
            local_phase = (phase - 0.15) / 0.2
            # Smooth extension from compressed state
            extension_progress = 0.5 * (1.0 + np.cos(np.pi * local_phase))
            compression_offset = self.compression_depth * extension_progress
            foot[2] += compression_offset
            foot[0] += inward_x_factor * 0.02 * extension_progress
            foot[1] += inward_y_factor * 0.02 * extension_progress

        # Phase 0.35-0.65: Aerial rotation (tucked)
        elif phase < 0.65:
            local_phase = (phase - 0.35) / 0.3
            # Smooth tuck with gradual entry and hold
            tuck_progress = 0.5 * (1.0 - np.cos(np.pi * min(local_phase / 0.5, 1.0)))
            foot[2] += self.tuck_height * tuck_progress
            foot[0] += inward_x_factor * self.tuck_inward * tuck_progress
            foot[1] += inward_y_factor * self.tuck_inward * tuck_progress

        # Phase 0.65-0.90: Pre-landing (BLENDED untuck and extend)
        elif phase < 0.90:
            local_phase = (phase - 0.65) / 0.25
            
            # Smooth untuck progression (cosine for smoothness)
            untuck_progress = 0.5 * (1.0 + np.cos(np.pi * (1.0 - local_phase)))
            
            # Smooth extension progression starting earlier, accelerating toward end
            # Use a shaped curve that starts gently and increases
            extension_progress = local_phase ** 1.5
            
            # Apply untuck (reducing tuck offsets)
            foot[2] += self.tuck_height * untuck_progress
            foot[0] += inward_x_factor * self.tuck_inward * untuck_progress
            foot[1] += inward_y_factor * self.tuck_inward * untuck_progress
            
            # Apply landing extension (moving downward from base position)
            foot[2] -= self.landing_extension * extension_progress

        # Phase 0.90-1.0: Landing absorption
        else:
            local_phase = (phase - 0.90) / 0.10
            
            # Maintain full extension until mid-phase, then gradually absorb
            if local_phase < 0.4:
                # Hold full extension during initial contact
                absorption_factor = 1.0
            else:
                # Gradual absorption with smooth curve
                absorption_local = (local_phase - 0.4) / 0.6
                absorption_factor = 1.0 - 0.25 * (0.5 * (1.0 - np.cos(np.pi * absorption_local)))
            
            current_extension = self.landing_extension * absorption_factor
            foot[2] -= current_extension

        return foot