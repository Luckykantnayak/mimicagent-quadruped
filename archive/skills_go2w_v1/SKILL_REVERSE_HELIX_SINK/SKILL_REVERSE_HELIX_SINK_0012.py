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
        
        # Compute nominal height using MINIMUM foot z-offset to ensure ALL feet can contact ground
        foot_z_offsets = [foot_pos[2] for foot_pos in initial_foot_positions_body.values()]
        lowest_foot_z = min(foot_z_offsets)
        
        # Set nominal height based on foot positions
        self.nominal_height = -lowest_foot_z
        
        # Motion parameters
        self.vx_backward = -0.25  # Sustained backward velocity (m/s)
        self.vz_descent_max = -0.06  # Peak downward velocity (m/s), conservative
        self.yaw_rate = 2.0 * np.pi * self.freq  # ~360° per cycle (rad/s)
        
        # Height descent parameters - conservative descent range
        self.min_height = self.nominal_height - 0.07  # Descend 7 cm
        self.min_height = max(self.min_height, 0.18)  # Safety floor
        self.height_descent_total = self.nominal_height - self.min_height
        
        # Leg spreading parameters
        self.lateral_spread_max = 0.02
        self.forward_spread_max = 0.015
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.nominal_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track integrated yaw for smooth control
        self.integrated_yaw = 0.0
        
        # Flag to track initialization phase
        self.initialization_complete = False

    def update_base_motion(self, phase, dt):
        """
        Update base with sustained backward velocity, constant yaw rate, and phase-modulated descent.
        """
        
        # Mark initialization as complete after first update
        if not self.initialization_complete and phase > 0.05:
            self.initialization_complete = True
            # Lock in the actual starting height as nominal
            self.nominal_height = self.root_pos[2]
            self.min_height = self.nominal_height - 0.07
            self.min_height = max(self.min_height, 0.18)
            self.height_descent_total = self.nominal_height - self.min_height
        
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
        
        CRITICAL FIX: Force ground contact during initialization phase by directly setting
        foot positions to reach ground from whatever base height is active.
        """
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # CRITICAL INITIALIZATION FIX: Force ground contact at start
        # During initialization (very early phase), directly set foot z to ensure ground contact
        if phase < 0.05:
            # Get current base height (may be set by framework, not our init)
            current_base_z = self.root_pos[2]
            
            # Calculate foot body z that will place foot at ground level (world z = 0)
            # foot_world_z = base_z + foot_body_z
            # We want foot_world_z = 0, so foot_body_z = -base_z
            target_foot_body_z = -current_base_z
            
            # Preserve xy position, override z for ground contact
            foot[2] = target_foot_body_z
            
            return foot
        
        # NORMAL OPERATION: After initialization, apply descent compensation
        current_height = self.root_pos[2]
        height_drop = self.nominal_height - current_height
        
        # Extend legs downward to maintain ground contact as base descends
        vertical_extension = -height_drop
        foot[2] += vertical_extension
        
        # Lateral spreading for stability at low heights
        if height_drop > 0.03:
            spread_factor = (height_drop - 0.03) / (self.height_descent_total - 0.03) if self.height_descent_total > 0.03 else 0.0
            spread_factor = np.clip(spread_factor, 0.0, 1.0)
            spread_factor = spread_factor ** 3
            
            if 'FL' in leg_name:
                foot[0] += self.forward_spread_max * spread_factor
                foot[1] += self.lateral_spread_max * spread_factor
            elif 'FR' in leg_name:
                foot[0] += self.forward_spread_max * spread_factor
                foot[1] -= self.lateral_spread_max * spread_factor
            elif 'RL' in leg_name:
                foot[0] -= self.forward_spread_max * spread_factor
                foot[1] += self.lateral_spread_max * spread_factor
            elif 'RR' in leg_name:
                foot[0] -= self.forward_spread_max * spread_factor
                foot[1] -= self.lateral_spread_max * spread_factor
        
        # Small cyclic adjustments for smooth tracking through rotation
        rotation_phase = (self.integrated_yaw / (2 * np.pi)) % 1.0
        cyclic_offset_x = 0.005 * np.sin(2 * np.pi * rotation_phase)
        cyclic_offset_y = 0.005 * np.cos(2 * np.pi * rotation_phase)
        
        foot[0] += cyclic_offset_x
        foot[1] += cyclic_offset_y
        
        return foot