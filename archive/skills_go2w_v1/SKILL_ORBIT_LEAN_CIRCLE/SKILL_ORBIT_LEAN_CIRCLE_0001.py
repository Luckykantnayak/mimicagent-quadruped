from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ORBIT_LEAN_CIRCLE_MotionGenerator(BaseMotionGenerator):
    """
    Circular banked turn with inward lean and alternating radial leg extension wave.
    
    - Base: constant forward velocity + constant yaw rate → circular path
    - Base roll: constant inward lean (toward turn center)
    - Legs: all four remain in contact; alternate between radially extended and tucked
      in a coordinated wave pattern (diagonal pairs phase-offset by 0.5)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0  # One full gait cycle per second
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state (world frame)
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Circular motion parameters
        self.vx = 0.8  # constant forward velocity (m/s)
        self.yaw_rate = 0.6  # constant yaw rate (rad/s) → circle radius ≈ vx / yaw_rate ≈ 1.33m
        
        # Inward lean angle (roll toward turn center for left turn, negative roll)
        self.roll_lean = -0.25  # radians (~14 degrees inward lean)
        
        # Apply constant roll lean to base orientation (one-time setup bias)
        self.root_quat = euler_to_quat(self.roll_lean, 0.0, 0.0)
        
        # Leg radial extension parameters
        self.radial_extension_amplitude = 0.12  # meters of lateral extension from nominal
        self.forward_shift_amplitude = 0.03  # slight forward/back shift during extension
        
        # Phase offsets for coordinated wave pattern
        # Front legs (FL, FR) act as outer legs initially (phase 0.0)
        # Rear legs (RL, RR) act as inner legs initially, then extend at phase 0.5
        self.phase_offsets = {
            'FL': 0.0,
            'FR': 0.0,
            'RL': 0.5,
            'RR': 0.5,
        }
        
        # Identify leg names dynamically
        self.fl_name = [n for n in leg_names if n.startswith('FL')][0]
        self.fr_name = [n for n in leg_names if n.startswith('FR')][0]
        self.rl_name = [n for n in leg_names if n.startswith('RL')][0]
        self.rr_name = [n for n in leg_names if n.startswith('RR')][0]
        
        self.phase_offsets = {
            self.fl_name: 0.0,
            self.fr_name: 0.0,
            self.rl_name: 0.5,
            self.rr_name: 0.5,
        }

    def update_base_motion(self, phase, dt):
        """
        Constant forward velocity and yaw rate produce circular path.
        Roll angle is constant (lean is maintained), so roll_rate = 0.
        """
        # Linear velocity: constant forward in world frame
        self.vel_world = np.array([self.vx, 0.0, 0.0])
        
        # Angular velocity: constant yaw rate, no roll or pitch rate
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate])
        
        # Integrate pose (maintains constant roll via quaternion integration)
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with radial extension wave.
        
        All feet remain in contact (z = base_z throughout).
        Legs alternate between extended (radially outward) and tucked (inward):
        - Phase 0.0-0.5: Front legs (FL, FR) extended, rear legs (RL, RR) tucked
        - Phase 0.5-1.0: Rear legs extended, front legs tucked
        
        Smooth sinusoidal transitions for continuous motion.
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Radial extension follows sinusoidal pattern:
        # leg_phase 0.0 → max extension
        # leg_phase 0.5 → max tucked (retracted)
        # leg_phase 1.0 → max extension again
        extension_factor = np.cos(2 * np.pi * leg_phase)  # +1 at 0, -1 at 0.5, +1 at 1.0
        
        # Lateral (y-axis) radial modulation
        # Left legs (FL, RL): positive y is outward
        # Right legs (FR, RR): negative y is outward
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            # Left side: extend in +y direction
            lateral_offset = self.radial_extension_amplitude * extension_factor
        else:
            # Right side: extend in -y direction
            lateral_offset = -self.radial_extension_amplitude * extension_factor
        
        foot[1] += lateral_offset
        
        # Slight forward/backward shift to enhance dynamic appearance
        # Front legs: shift forward when extended
        # Rear legs: shift backward when extended
        if leg_name.startswith('FL') or leg_name.startswith('FR'):
            foot[0] += self.forward_shift_amplitude * extension_factor
        else:
            foot[0] -= self.forward_shift_amplitude * extension_factor
        
        # Z remains at base contact height (no lifting)
        # foot[2] unchanged from base_feet_pos_body
        
        return foot