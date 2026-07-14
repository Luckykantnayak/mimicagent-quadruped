from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_WINDSHIELD_WIPER_YAW_MotionGenerator(BaseMotionGenerator):
    """
    Windshield wiper yaw oscillation: asymmetric in-place rotation.
    
    Phase structure:
      [0.0, 0.6]: Slow clockwise yaw sweep (~50 degrees accumulated)
      [0.6, 0.7]: Fast counterclockwise yaw return (reverse ~50 degrees)
      [0.7, 1.0]: Pause phase (zero yaw rate, stabilization)
    
    All four legs remain in continuous ground contact throughout.
    Legs adjust their body-frame positions to accommodate base yaw rotation
    while maintaining stationary world-frame foot positions (kinematic in-place rotation).
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Cycle frequency (Hz)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Determine nominal base height from initial foot positions
        avg_foot_z_body = np.mean([v[2] for v in initial_foot_positions_body.values()])
        self.nominal_base_height = -avg_foot_z_body
        if self.nominal_base_height < 0.4:
            self.nominal_base_height = 0.4
        
        # Yaw motion parameters
        self.target_yaw_displacement = np.deg2rad(50.0)
        
        # Phase boundaries
        self.sweep_end = 0.6
        self.return_end = 0.7
        self.pause_end = 1.0
        
        # Compute yaw rates to achieve target displacement
        sweep_duration_per_cycle = self.sweep_end
        self.yaw_rate_sweep = self.target_yaw_displacement / (sweep_duration_per_cycle / self.freq)
        
        return_duration_per_cycle = (self.return_end - self.sweep_end)
        self.yaw_rate_return = -self.target_yaw_displacement / (return_duration_per_cycle / self.freq)
        
        # Base height modulation to relieve lateral joint strain at peak yaw
        self.max_height_increase = 0.02
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.nominal_base_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Establish fixed world-frame foot anchor points at initialization
        # These are the world positions where feet should remain grounded
        self.world_foot_anchors = {}
        for leg_name in self.leg_names:
            foot_body = self.base_feet_pos_body[leg_name]
            # Transform initial body-frame position to world frame
            # At init, base is at origin with no rotation
            foot_world = self.root_pos + foot_body
            # Pin feet to ground plane
            foot_world[2] = 0.0
            self.world_foot_anchors[leg_name] = foot_world.copy()

    def update_base_motion(self, phase, dt):
        """
        Update base yaw according to phase-dependent yaw rate profile.
        No translation: robot remains in place.
        Base height modulated to reduce joint strain during peak rotation.
        """
        # Determine yaw rate based on phase with smooth transitions
        if phase < self.sweep_end:
            # Slow clockwise sweep
            phase_progress = phase / self.sweep_end
            smoothing = self._smooth_step(phase_progress)
            yaw_rate = self.yaw_rate_sweep * smoothing
            
        elif phase < self.return_end:
            # Fast counterclockwise return
            phase_progress = (phase - self.sweep_end) / (self.return_end - self.sweep_end)
            smoothing = self._smooth_step(phase_progress)
            yaw_rate = self.yaw_rate_return * smoothing
            
        else:
            # Pause phase: zero yaw rate
            yaw_rate = 0.0
        
        # Compute base height adjustment based on absolute yaw magnitude
        height_adjustment = self._compute_height_adjustment(phase)
        
        # Set velocity commands (zero translation, yaw rotation only)
        self.vel_world = np.array([0.0, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Set base height to nominal height plus modulation
        self.root_pos[2] = self.nominal_base_height + height_adjustment

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame by transforming world-frame anchor point.
        
        Feet remain at fixed world positions (on ground plane z=0).
        Body-frame representation changes as base yaws to maintain world-frame stationarity.
        """
        # Get the fixed world-frame anchor point for this foot
        foot_world = self.world_foot_anchors[leg_name].copy()
        
        # Transform world-frame foot position to current body frame
        # foot_body = R^T * (foot_world - root_pos)
        # where R is rotation matrix from quaternion
        
        # Compute relative position in world frame
        foot_rel_world = foot_world - self.root_pos
        
        # Convert quaternion to rotation matrix and transpose for inverse rotation
        # For body frame transformation, we need R^T * foot_rel_world
        quat = self.root_quat
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Rotation matrix from quaternion (world to body is transpose)
        # Build R^T directly (body to world transposed = world to body)
        R_T = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y + w*z),     2*(x*z - w*y)],
            [    2*(x*y - w*z), 1 - 2*(x*x + z*z),     2*(y*z + w*x)],
            [    2*(x*z + w*y),     2*(y*z - w*x), 1 - 2*(x*x + y*y)]
        ])
        
        # Transform to body frame
        foot_body = R_T @ foot_rel_world
        
        return foot_body

    def _compute_height_adjustment(self, phase):
        """
        Compute base height adjustment to relieve joint strain during peak rotation.
        Height increases smoothly during maximum yaw displacement phases.
        Returns absolute height adjustment to be added to nominal height.
        """
        # Compute normalized yaw magnitude based on phase
        if phase < self.sweep_end:
            # During sweep: yaw grows from 0 to 1
            phase_progress = phase / self.sweep_end
            yaw_normalized = self._smooth_step(phase_progress)
            
        elif phase < self.return_end:
            # During return: yaw decreases from 1 to 0
            phase_progress = (phase - self.sweep_end) / (self.return_end - self.sweep_end)
            yaw_normalized = 1.0 - self._smooth_step(phase_progress)
            
        else:
            # During pause: no yaw offset
            yaw_normalized = 0.0
        
        # Smooth symmetric height increase: peaks at mid-sweep and mid-return
        # Use bell curve to ensure smooth return to zero at pause
        height_adjustment = self.max_height_increase * np.sin(yaw_normalized * np.pi)
        
        return height_adjustment

    def _smooth_step(self, t):
        """
        Smooth step function for reducing jerk at phase transitions.
        Uses smoothstep: 3t^2 - 2t^3
        Input t in [0, 1], output smoothly interpolates from 0 to 1.
        """
        t = np.clip(t, 0.0, 1.0)
        return 3.0 * t**2 - 2.0 * t**3