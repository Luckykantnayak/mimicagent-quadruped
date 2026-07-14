from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_LATERAL_COIL_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Lateral coil spin: in-place yaw rotation with progressive leg retraction.
    
    - All four legs remain in continuous ground contact (stance)
    - Legs retract radially inward toward body centerline over phase [0,1]
    - Base rises to maintain stability as stance narrows
    - Yaw rate increases progressively through the cycle
    - One full cycle completes ~360° rotation with maximum coil at phase=1.0
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (body frame, initial wide stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Store initial foot positions in world frame (on ground)
        self.initial_feet_world = {}
        for leg_name, pos_body in self.base_feet_pos_body.items():
            # Assume initial base at origin with identity orientation
            self.initial_feet_world[leg_name] = pos_body.copy()
            self.initial_feet_world[leg_name][2] = 0.0  # Enforce ground level
        
        # Compute radial direction for each leg in horizontal plane
        self.leg_radial_dirs = {}
        for leg_name, pos in self.base_feet_pos_body.items():
            horizontal_pos = np.array([pos[0], pos[1], 0.0])
            radial_dist = np.linalg.norm(horizontal_pos)
            if radial_dist > 1e-6:
                self.leg_radial_dirs[leg_name] = horizontal_pos / radial_dist
            else:
                self.leg_radial_dirs[leg_name] = np.array([1.0, 0.0, 0.0])
        
        # Retraction parameters: calibrated to match skill milestones
        # 30% at phase 0.5, 60% at phase 0.75, 65% max at phase 1.0
        self.max_retraction_ratio = 0.65
        
        # Base rise parameters: reduced to prevent joint limit violations
        self.max_base_rise = 0.035  # Reduced from 0.06 to 0.035 meters
        
        # Yaw rotation parameters
        self.yaw_rate_initial = 2.0  # rad/s at phase 0
        self.yaw_rate_final = 4.0    # rad/s at phase 1 (doubled)
        
        # Base state (no longer incrementally integrated)
        self.t = 0.0

    def compute_base_height(self, phase):
        """
        Compute base height directly from phase.
        Peak shifted slightly later to align with advanced retraction.
        """
        # Sinusoidal rise peaking at phase ~0.6, returning to 0 at phase 1.0
        # Using sin(1.1 * pi * phase) to shift peak slightly right
        height_profile = np.sin(1.1 * np.pi * phase)
        if phase > 0.909:  # After peak, ensure smooth return to zero
            height_profile = np.sin(np.pi * phase)
        return self.max_base_rise * height_profile

    def compute_yaw_angle(self, phase):
        """
        Compute total yaw rotation angle directly from phase.
        Yaw rate increases linearly from yaw_rate_initial to yaw_rate_final.
        
        yaw_rate(t) = yaw_rate_initial + (yaw_rate_final - yaw_rate_initial) * phase
        Integrating over phase from 0 to p:
        yaw(p) = yaw_rate_initial * p + 0.5 * (yaw_rate_final - yaw_rate_initial) * p^2
        
        Convert phase to time: phase = freq * t, so t = phase / freq
        yaw(phase) = (yaw_rate_initial * phase + 0.5 * (yaw_rate_final - yaw_rate_initial) * phase^2) / freq
        """
        yaw_angle = (self.yaw_rate_initial * phase + 
                     0.5 * (self.yaw_rate_final - self.yaw_rate_initial) * phase**2) / self.freq
        return yaw_angle

    def compute_retraction_ratio(self, phase):
        """
        Compute retraction ratio matching skill description milestones:
        - Phase 0.25: ~10% retraction (wide stance)
        - Phase 0.5: 30% retraction
        - Phase 0.75: 60% retraction
        - Phase 1.0: 65% retraction (maximum)
        
        Using piecewise polynomial interpolation for smooth progression.
        """
        if phase < 0.5:
            # Linear ramp to 30% at phase 0.5
            t_norm = phase / 0.5
            return 0.3 * self.max_retraction_ratio * t_norm
        elif phase < 0.75:
            # Accelerating curve from 30% to 60% over phase 0.5 to 0.75
            t_norm = (phase - 0.5) / 0.25
            # Smooth interpolation from 0.3 to 0.6
            start_ratio = 0.3
            end_ratio = 0.6
            blend = 3 * t_norm**2 - 2 * t_norm**3  # Smooth step
            return self.max_retraction_ratio * (start_ratio + (end_ratio - start_ratio) * blend)
        else:
            # Final tightening from 60% to 65% (max) over phase 0.75 to 1.0
            t_norm = (phase - 0.75) / 0.25
            start_ratio = 0.6
            end_ratio = 1.0  # 100% of max_retraction_ratio (0.65)
            blend = 3 * t_norm**2 - 2 * t_norm**3
            return self.max_retraction_ratio * (start_ratio + (end_ratio - start_ratio) * blend)

    def compute_base_state(self, phase):
        """
        Compute base position and orientation directly from phase.
        No incremental integration - all quantities computed as closed-form functions.
        """
        # Base position: no horizontal translation, only vertical rise
        root_pos = np.array([0.0, 0.0, self.compute_base_height(phase)])
        
        # Base orientation: pure yaw rotation around z-axis
        yaw_angle = self.compute_yaw_angle(phase)
        root_quat = np.array([
            np.cos(yaw_angle / 2.0),
            0.0,
            0.0,
            np.sin(yaw_angle / 2.0)
        ])
        
        return root_pos, root_quat

    def compute_base_velocities(self, phase):
        """
        Compute base velocities directly from phase derivatives.
        """
        # Vertical velocity: derivative of height profile
        # d/dphase [max_rise * sin(1.1 * pi * phase)] = max_rise * 1.1 * pi * cos(1.1 * pi * phase)
        # dz/dt = dz/dphase * dphase/dt = dz/dphase * freq
        if phase > 0.909:
            vz = self.max_base_rise * np.pi * np.cos(np.pi * phase) * self.freq
        else:
            vz = self.max_base_rise * 1.1 * np.pi * np.cos(1.1 * np.pi * phase) * self.freq
        
        # Yaw rate: current instantaneous yaw rate
        yaw_rate = self.yaw_rate_initial + (self.yaw_rate_final - self.yaw_rate_initial) * phase
        
        vel_world = np.array([0.0, 0.0, vz])
        omega_world = np.array([0.0, 0.0, yaw_rate])
        
        return vel_world, omega_world

    def compute_foot_position_body_frame(self, leg_name, phase, root_pos, root_quat):
        """
        Compute foot position in body frame with radial retraction while maintaining ground contact.
        
        Each foot moves radially inward toward the body centerline in the horizontal plane.
        Vertical position is computed to maintain z=0 in world frame (continuous ground contact).
        """
        # Get initial foot position in world frame
        initial_world = self.initial_feet_world[leg_name].copy()
        
        # Compute current retraction ratio
        retraction_ratio = self.compute_retraction_ratio(phase)
        
        # Compute retracted horizontal position in world frame
        horizontal_initial = np.array([initial_world[0], initial_world[1]])
        horizontal_retracted = horizontal_initial * (1.0 - retraction_ratio)
        
        # Foot position in world frame (on ground)
        foot_world = np.array([
            horizontal_retracted[0],
            horizontal_retracted[1],
            0.0  # Always on ground
        ])
        
        # Transform to body frame
        # foot_body = R^T * (foot_world - root_pos)
        foot_relative = foot_world - root_pos
        
        # Convert quaternion to rotation matrix and apply inverse rotation
        qw, qx, qy, qz = root_quat
        
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        # Apply inverse rotation (transpose of R)
        foot_body = R.T @ foot_relative
        
        return foot_body

    def get_state(self, t):
        """
        Main interface: compute full robot state at time t.
        All state quantities computed directly from phase - no incremental integration.
        """
        self.t = t
        phase = (self.freq * t) % 1.0
        
        # Compute base state directly from phase
        root_pos, root_quat = self.compute_base_state(phase)
        
        # Compute base velocities directly from phase
        vel_world, omega_world = self.compute_base_velocities(phase)
        
        # Compute foot positions in body frame
        foot_positions_body = {}
        for leg_name in self.leg_names:
            foot_positions_body[leg_name] = self.compute_foot_position_body_frame(
                leg_name, phase, root_pos, root_quat
            )
        
        return {
            'root_position': root_pos.copy(),
            'root_quaternion': root_quat.copy(),
            'foot_positions_body_frame': foot_positions_body,
            'root_linear_velocity': vel_world.copy(),
            'root_angular_velocity': omega_world.copy()
        }