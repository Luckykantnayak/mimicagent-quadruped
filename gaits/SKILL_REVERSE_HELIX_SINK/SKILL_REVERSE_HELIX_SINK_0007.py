from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERSE_HELIX_SINK_MotionGenerator(BaseMotionGenerator):
    """
    Reverse Helix Sink: Robot executes a continuous backward spiral descent.
    
    Kinematic motion combining:
    - Sustained backward velocity (negative vx)
    - Counter-clockwise yaw rotation (~360° per cycle)
    - Gradual downward velocity (negative vz) from nominal to minimum height
    
    All four legs remain in continuous ground contact throughout.
    Legs adjust body-frame positions to compensate for base rotation, translation, and descent.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slow cycle for smooth helical descent
        
        # Base foot positions (body frame at nominal height)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.vx_backward = -0.4  # Sustained backward velocity (m/s)
        self.vz_descent_max = -0.15  # Peak downward velocity (m/s)
        self.yaw_rate = 2.0 * np.pi * self.freq  # ~360° per cycle (rad/s)
        
        # Height descent parameters
        self.nominal_height = 0.3  # Starting height
        self.min_height = 0.12  # Target minimum height
        self.height_descent_total = self.nominal_height - self.min_height
        
        # Leg spreading parameters (wider stance at lower heights)
        self.lateral_spread_max = 0.06  # Maximum outward spread per leg
        self.forward_spread_max = 0.04  # Maximum forward/backward spread
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.nominal_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track integrated yaw and height for smooth control
        self.integrated_yaw = 0.0
        self.current_height = self.nominal_height

    def update_base_motion(self, phase, dt):
        """
        Update base with sustained backward velocity, constant yaw rate, and phase-modulated descent.
        
        Velocity profile:
        - vx: constant backward
        - vy: zero
        - vz: sinusoidal modulation for smooth descent (peak at phase ~0.25-0.5)
        - yaw_rate: constant counter-clockwise rotation
        """
        
        # Linear velocity - backward with phase-modulated descent
        vx = self.vx_backward
        vy = 0.0
        
        # Descent velocity: sinusoidal envelope to smooth start/end, peak mid-cycle
        # Use sin^2 for smooth ramp up and down
        descent_envelope = np.sin(np.pi * phase) ** 2
        vz = self.vz_descent_max * descent_envelope
        
        # Clamp height at minimum
        if self.current_height <= self.min_height:
            vz = 0.0
        
        self.vel_world = np.array([vx, vy, vz])
        
        # Angular velocity - constant counter-clockwise yaw
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = self.yaw_rate
        
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Track current height and integrated yaw for leg compensation
        self.current_height = self.root_pos[2]
        self.integrated_yaw += yaw_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute body-frame foot position to maintain ground contact during helical descent.
        
        Compensation strategy:
        1. Vertical extension: legs extend downward as base height drops
        2. Rotational drift: feet drift in body frame opposite to yaw rotation
        3. Translational drift: feet drift forward opposite to backward base motion
        4. Lateral spreading: legs spread outward at lower heights for stability
        """
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # 1. Vertical extension based on height descent
        height_drop = self.nominal_height - self.current_height
        vertical_extension = -height_drop  # Extend downward (negative z in body frame)
        foot[2] += vertical_extension
        
        # 2. Lateral spreading for stability at low heights
        # Spread increases as height decreases
        spread_factor = height_drop / self.height_descent_total if self.height_descent_total > 0 else 0.0
        spread_factor = np.clip(spread_factor, 0.0, 1.0)
        
        # Determine leg side and position for spreading
        if leg_name.startswith('FL'):
            # Front left: spread left (+y) and forward (+x)
            foot[0] += self.forward_spread_max * spread_factor
            foot[1] += self.lateral_spread_max * spread_factor
        elif leg_name.startswith('FR'):
            # Front right: spread right (-y) and forward (+x)
            foot[0] += self.forward_spread_max * spread_factor
            foot[1] -= self.lateral_spread_max * spread_factor
        elif leg_name.startswith('RL'):
            # Rear left: spread left (+y) and backward (-x)
            foot[0] -= self.forward_spread_max * spread_factor
            foot[1] += self.lateral_spread_max * spread_factor
        elif leg_name.startswith('RR'):
            # Rear right: spread right (-y) and backward (-x)
            foot[0] -= self.forward_spread_max * spread_factor
            foot[1] -= self.lateral_spread_max * spread_factor
        
        # 3. Rotational compensation in body frame
        # As base rotates counter-clockwise (yaw increases), feet appear to rotate clockwise in body frame
        # This is handled implicitly by the body-to-world transform, but we add small cyclic adjustments
        # to maintain smooth tracking through the rotation
        rotation_phase = (self.integrated_yaw / (2 * np.pi)) % 1.0
        
        # Small cyclic adjustment to maintain ground contact through rotation
        # (compensates for any leg kinematic coupling effects)
        cyclic_offset_x = 0.01 * np.sin(2 * np.pi * rotation_phase)
        cyclic_offset_y = 0.01 * np.cos(2 * np.pi * rotation_phase)
        
        foot[0] += cyclic_offset_x
        foot[1] += cyclic_offset_y
        
        # 4. Forward drift compensation for backward base motion
        # As base moves backward, feet drift forward in body frame to maintain world position
        # Approximate integrated backward displacement over phase
        backward_drift = self.vx_backward * phase / self.freq
        foot[0] -= backward_drift * 0.1  # Scale factor for smooth compensation
        
        return foot