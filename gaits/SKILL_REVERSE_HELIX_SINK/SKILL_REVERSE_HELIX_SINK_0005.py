from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_REVERSE_HELIX_SINK_MotionGenerator(BaseMotionGenerator):
    """
    Reverse Helical Descent Motion Generator.
    
    The robot executes a continuous backward translation while rotating 
    counter-clockwise and descending in height, tracing a helical spiral 
    path downward over one full phase cycle. All four feet maintain 
    ground contact throughout.
    
    - Backward velocity: constant negative vx
    - Yaw rotation: constant positive yaw_rate (360° per cycle)
    - Vertical descent: time-varying negative vz (peaks mid-cycle, tapers to zero)
    - All legs remain in stance, compressing to lower base while repositioning
      in body frame to track the rotating base
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0  # One full helical cycle per phase period
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.backward_velocity = -0.4  # Constant backward velocity (m/s)
        self.yaw_rate_total = 2 * np.pi  # 360 degrees per cycle
        self.descent_total = 0.15  # Total vertical descent over cycle (m)
        self.min_height_offset = -0.15  # Maximum descent from initial height
        
        # Leg compression and spread parameters
        self.max_compression = 0.15  # Maximum vertical leg compression (m)
        self.lateral_spread_max = 0.05  # Maximum lateral spread for stability (m)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base using constant backward velocity, constant yaw rate,
        and phase-varying descent velocity to create helical trajectory.
        """
        # Constant backward velocity
        vx = self.backward_velocity
        
        # Constant yaw rate to complete 360° over one cycle
        # yaw_rate = yaw_rate_total * freq
        yaw_rate = self.yaw_rate_total * self.freq
        
        # Time-varying descent velocity
        # Peak descent rate at phase 0.25-0.5, taper to zero by phase 0.75-1.0
        if phase < 0.25:
            # Smoothly ramp up descent rate (phase 0.0 -> 0.25)
            descent_progress = phase / 0.25
            vz_scale = np.sin(np.pi * descent_progress / 2)  # Smooth ease-in
        elif phase < 0.5:
            # Peak descent rate (phase 0.25 -> 0.5)
            vz_scale = 1.0
        elif phase < 0.75:
            # Taper descent rate (phase 0.5 -> 0.75)
            descent_progress = (phase - 0.5) / 0.25
            vz_scale = np.cos(np.pi * descent_progress / 2)  # Smooth ease-out
        else:
            # Minimal/zero descent at minimum height (phase 0.75 -> 1.0)
            vz_scale = 0.0
        
        # Scale descent velocity to achieve total descent over cycle
        # Average vz needed: descent_total * freq
        # Peak vz (assuming sinusoidal-like profile): ~2 * average
        vz_peak = -2.0 * self.descent_total * self.freq
        vz = vz_peak * vz_scale
        
        # Set world-frame velocity commands
        self.vel_world = np.array([vx, 0.0, vz])
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
        Compute foot position in body frame.
        
        All feet remain in stance (grounded). To maintain contact while the base
        rotates and descends:
        - Legs compress vertically (z becomes more negative) as phase progresses
        - Legs spread slightly laterally for stability at low height
        - Body-frame x,y positions shift to counteract yaw rotation effect
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Vertical compression profile
        # Compress smoothly from phase 0 to phase 0.75, maintain at phase 0.75-1.0
        if phase < 0.75:
            compression_progress = phase / 0.75
            # Smooth compression curve (ease-in-out)
            compression_factor = 0.5 * (1 - np.cos(np.pi * compression_progress))
        else:
            compression_factor = 1.0
        
        compression = self.max_compression * compression_factor
        foot[2] -= compression  # Lower foot in body frame (more negative z)
        
        # Lateral spread for stability (peaks at maximum compression)
        # Front legs spread forward-outward, rear legs spread backward-outward
        spread_factor = compression_factor
        lateral_spread = self.lateral_spread_max * spread_factor
        
        if leg_name.startswith('FL'):
            # Front-left: spread slightly forward and left
            foot[0] += 0.3 * lateral_spread
            foot[1] += lateral_spread
        elif leg_name.startswith('FR'):
            # Front-right: spread slightly forward and right
            foot[0] += 0.3 * lateral_spread
            foot[1] -= lateral_spread
        elif leg_name.startswith('RL'):
            # Rear-left: spread slightly backward and left
            foot[0] -= 0.3 * lateral_spread
            foot[1] += lateral_spread
        elif leg_name.startswith('RR'):
            # Rear-right: spread slightly backward and right
            foot[0] -= 0.3 * lateral_spread
            foot[1] -= lateral_spread
        
        # Compensate for yaw rotation in body frame
        # As base rotates counter-clockwise, body-frame foot positions must
        # rotate clockwise (negative direction) to maintain world-frame ground contact
        yaw_angle = -self.yaw_rate_total * phase  # Negative for clockwise compensation
        cos_yaw = np.cos(yaw_angle)
        sin_yaw = np.sin(yaw_angle)
        
        # Rotate foot position in body frame x-y plane
        base_foot = self.base_feet_pos_body[leg_name].copy()
        x_rotated = base_foot[0] * cos_yaw - base_foot[1] * sin_yaw
        y_rotated = base_foot[0] * sin_yaw + base_foot[1] * cos_yaw
        
        # Apply rotation to base position, then add compression and spread
        foot[0] = x_rotated
        foot[1] = y_rotated
        foot[2] = self.base_feet_pos_body[leg_name][2] - compression
        
        # Re-apply lateral spread after rotation
        if leg_name.startswith('FL'):
            foot[0] += 0.3 * lateral_spread
            foot[1] += lateral_spread
        elif leg_name.startswith('FR'):
            foot[0] += 0.3 * lateral_spread
            foot[1] -= lateral_spread
        elif leg_name.startswith('RL'):
            foot[0] -= 0.3 * lateral_spread
            foot[1] += lateral_spread
        elif leg_name.startswith('RR'):
            foot[0] -= 0.3 * lateral_spread
            foot[1] -= lateral_spread
        
        return foot