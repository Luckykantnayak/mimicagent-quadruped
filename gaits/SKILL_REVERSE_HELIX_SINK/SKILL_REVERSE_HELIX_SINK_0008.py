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
        
        # Compute initial base height from foot positions to ensure ground contact at start
        # Feet should be at z=0 in world frame, so base height = -foot_z_body
        avg_foot_z_body = np.mean([pos[2] for pos in initial_foot_positions_body.values()])
        self.nominal_height = -avg_foot_z_body  # Start height ensuring feet at ground
        self.nominal_height = max(self.nominal_height, 0.25)  # Safety minimum
        
        # Motion parameters
        self.vx_backward = -0.3  # Sustained backward velocity (m/s), reduced for stability
        self.vz_descent_max = -0.08  # Peak downward velocity (m/s), reduced for controllability
        self.yaw_rate = 2.0 * np.pi * self.freq  # ~360° per cycle (rad/s)
        
        # Height descent parameters
        self.min_height = self.nominal_height - 0.15  # Descent by 15 cm total
        self.min_height = max(self.min_height, 0.18)  # Safety floor to prevent ground penetration
        self.height_descent_total = self.nominal_height - self.min_height
        
        # Leg spreading parameters (wider stance at lower heights)
        self.lateral_spread_max = 0.04  # Maximum outward spread per leg
        self.forward_spread_max = 0.03  # Maximum forward/backward spread
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.nominal_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track integrated yaw for smooth control
        self.integrated_yaw = 0.0

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
        descent_envelope = np.sin(np.pi * phase) ** 2
        vz = self.vz_descent_max * descent_envelope
        
        # Proactive height clamping: prevent descent below minimum height
        current_height = self.root_pos[2]
        predicted_height = current_height + vz * dt
        
        if predicted_height < self.min_height:
            # Clamp velocity to exactly reach minimum height, or zero if already there
            if current_height > self.min_height:
                vz = (self.min_height - current_height) / dt
            else:
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
        
        # Post-integration safety clamp: hard constraint on minimum height
        self.root_pos[2] = max(self.root_pos[2], self.min_height)
        
        # Track integrated yaw for leg compensation
        self.integrated_yaw += yaw_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute body-frame foot position to maintain ground contact during helical descent.
        
        Compensation strategy:
        1. Vertical extension: legs extend downward as base height drops
        2. Lateral spreading: legs spread outward at lower heights for stability
        3. Rotational/translational drift compensation: small cyclic adjustments
        """
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # 1. Vertical extension based on height descent
        current_height = self.root_pos[2]
        height_drop = self.nominal_height - current_height
        
        # Extend legs downward to maintain ground contact as base sinks
        # Foot z in body frame becomes more negative (extends down)
        vertical_extension = -height_drop * 0.8  # Scale to prevent over-extension
        foot[2] += vertical_extension
        
        # 2. Lateral spreading for stability at low heights
        spread_factor = height_drop / self.height_descent_total if self.height_descent_total > 0 else 0.0
        spread_factor = np.clip(spread_factor, 0.0, 1.0)
        
        # Smooth spreading with cubic easing
        spread_factor = spread_factor ** 2
        
        # Determine leg side and position for spreading
        if 'FL' in leg_name:
            # Front left: spread left (+y) and slightly forward (+x)
            foot[0] += self.forward_spread_max * spread_factor
            foot[1] += self.lateral_spread_max * spread_factor
        elif 'FR' in leg_name:
            # Front right: spread right (-y) and slightly forward (+x)
            foot[0] += self.forward_spread_max * spread_factor
            foot[1] -= self.lateral_spread_max * spread_factor
        elif 'RL' in leg_name:
            # Rear left: spread left (+y) and slightly backward (-x)
            foot[0] -= self.forward_spread_max * spread_factor
            foot[1] += self.lateral_spread_max * spread_factor
        elif 'RR' in leg_name:
            # Rear right: spread right (-y) and slightly backward (-x)
            foot[0] -= self.forward_spread_max * spread_factor
            foot[1] -= self.lateral_spread_max * spread_factor
        
        # 3. Small cyclic adjustments for smooth tracking through rotation
        rotation_phase = (self.integrated_yaw / (2 * np.pi)) % 1.0
        
        # Gentle sinusoidal compensation to maintain ground contact through yaw
        cyclic_offset_x = 0.008 * np.sin(2 * np.pi * rotation_phase)
        cyclic_offset_y = 0.008 * np.cos(2 * np.pi * rotation_phase)
        
        foot[0] += cyclic_offset_x
        foot[1] += cyclic_offset_y
        
        return foot