from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CROSSOVER_WAVE_SKATE_MotionGenerator(BaseMotionGenerator):
    """
    Crossover wave skating gait: rhythmic lateral leg sweeps with traveling wave dynamics.
    
    - Front and rear leg pairs operate in anti-phase (0.5 offset)
    - Legs sweep inward (unload/slide) then outward (carve/push)
    - Base rolls and yaws in sync to amplify wave energy transfer
    - All four wheels remain in continuous ground contact
    - Forward propulsion via synchronized outward carves
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Wave cycle frequency (Hz)
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.lateral_sweep_amplitude = 0.10  # Lateral crossover distance
        self.longitudinal_amplitude = 0.05   # Forward/backward component during sweep
        
        # Base motion parameters
        self.vx_base = 0.8  # Base forward velocity
        self.vx_modulation = 0.2  # Forward velocity modulation amplitude
        self.roll_amplitude = 0.08  # Roll oscillation amplitude (rad)
        self.yaw_amplitude = 0.12   # Yaw oscillation amplitude (rad)
        
        # Phase offsets: front pair at 0.0, rear pair at 0.5 (anti-phase)
        self.phase_offsets = {
            leg_names[0]: 0.0,  # FL
            leg_names[1]: 0.0,  # FR
            leg_names[2]: 0.5,  # RL
            leg_names[3]: 0.5,  # RR
        }
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track current roll angle extracted from quaternion
        self.current_roll = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base with forward velocity and synchronized roll/yaw oscillations.
        
        Roll: shifts weight between left/right to facilitate leg unloading
        Yaw: aligns base with carving direction to redirect momentum
        Forward velocity: modulated to reinforce propulsion during carve phases
        """
        # Forward velocity with periodic modulation
        vx = self.vx_base + self.vx_modulation * np.cos(2 * np.pi * phase)
        
        # Roll oscillation
        roll_rate = -2 * np.pi * self.freq * self.roll_amplitude * np.sin(2 * np.pi * phase)
        
        # Yaw oscillation
        yaw_rate = 2 * np.pi * self.freq * self.yaw_amplitude * np.cos(2 * np.pi * phase)
        
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Extract actual roll angle from integrated quaternion
        # For quaternion [w, x, y, z], roll (rotation about x-axis) can be extracted via:
        # roll = atan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2))
        w, x, y, z = self.root_quat
        self.current_roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for traveling wave crossover motion.
        
        Phase cycle per leg:
        - [0.0, 0.25]: sweep inward (toward centerline) + slightly rearward
        - [0.25, 0.5]: transition outward + slightly forward
        - [0.5, 0.75]: carve outward (away from centerline) + rearward (push)
        - [0.75, 1.0]: transition inward + forward (return to neutral)
        
        Front legs (FL, FR): phase offset 0.0
        Rear legs (RL, RR): phase offset 0.5 (anti-phase)
        
        All wheels maintain continuous ground contact with vertical compensation for roll.
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine lateral sign based on leg side
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            # Left legs: base position has positive y
            lateral_sign = 1.0
        else:
            # Right legs: base position has negative y
            lateral_sign = -1.0
        
        # Use BASE lateral position (before sweep) for roll compensation
        base_lateral_y = foot[1]
        
        # Vertical compensation for roll-induced height variation
        # When body rolls about x-axis (forward direction), a foot at lateral position y experiences:
        # - Vertical displacement in world frame: delta_z_world = y * sin(roll)
        # - To maintain ground contact (world z = constant), body frame z must compensate
        # For small angles: when roll > 0 (left side down), left legs (positive y) need to move UP in body frame
        # Correct formula: add compensation that counteracts the rotation-induced world-frame height change
        vertical_compensation = base_lateral_y * (1.0 - np.cos(self.current_roll))
        
        # Apply vertical compensation first
        foot[2] += vertical_compensation
        
        # Smooth sinusoidal lateral sweep
        lateral_offset = lateral_sign * self.lateral_sweep_amplitude * np.cos(2 * np.pi * (leg_phase - 0.25))
        
        # Longitudinal component
        longitudinal_offset = -self.longitudinal_amplitude * np.sin(4 * np.pi * leg_phase)
        
        foot[0] += longitudinal_offset
        foot[1] += lateral_offset
        
        return foot