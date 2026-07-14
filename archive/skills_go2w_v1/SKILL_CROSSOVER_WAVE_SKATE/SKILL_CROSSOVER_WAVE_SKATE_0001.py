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
        self.lateral_sweep_amplitude = 0.12  # Lateral crossover distance
        self.longitudinal_amplitude = 0.05   # Forward/backward component during sweep
        self.vertical_modulation = 0.02      # Small vertical oscillation for weight shift
        
        # Base motion parameters
        self.vx_base = 0.8  # Base forward velocity
        self.vx_modulation = 0.2  # Forward velocity modulation amplitude
        self.roll_amplitude = 0.15  # Roll oscillation amplitude (rad)
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

    def update_base_motion(self, phase, dt):
        """
        Update base with forward velocity and synchronized roll/yaw oscillations.
        
        Roll: shifts weight between left/right to facilitate leg unloading
        Yaw: aligns base with carving direction to redirect momentum
        Forward velocity: modulated to reinforce propulsion during carve phases
        """
        # Forward velocity with periodic modulation (boost during rear carve: phase 0-0.25)
        vx = self.vx_base + self.vx_modulation * np.cos(2 * np.pi * phase)
        
        # Roll oscillation: negative at phase 0 (weight to rear-right), positive at phase 0.5 (weight to front-left)
        roll_rate = -2 * np.pi * self.freq * self.roll_amplitude * np.sin(2 * np.pi * phase)
        
        # Yaw oscillation: positive at phase 0 (align with rear carve), negative at phase 0.5 (align with front carve)
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
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine lateral sign based on leg side (left = positive y, right = negative y)
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            # Left legs: base position has positive y
            lateral_sign = 1.0
        else:
            # Right legs: base position has negative y
            lateral_sign = -1.0
        
        # Smooth sinusoidal lateral sweep: inward at phase 0.0-0.5, outward at phase 0.5-1.0
        # Use cosine so that at leg_phase=0.25 we're at minimum (inward), at leg_phase=0.75 we're at maximum (outward)
        lateral_offset = lateral_sign * self.lateral_sweep_amplitude * np.cos(2 * np.pi * (leg_phase - 0.25))
        
        # Longitudinal component: rearward during inward sweep (0-0.25) and outward carve (0.5-0.75)
        # Forward during transitions (0.25-0.5 and 0.75-1.0)
        longitudinal_offset = -self.longitudinal_amplitude * np.sin(4 * np.pi * leg_phase)
        
        # Small vertical modulation to represent weight shift (lower when unloaded, higher when loaded)
        # Lowest at inward sweep peak (phase 0.25), highest at outward carve peak (phase 0.75)
        vertical_offset = self.vertical_modulation * np.sin(2 * np.pi * (leg_phase - 0.25))
        
        foot[0] += longitudinal_offset
        foot[1] += lateral_offset
        foot[2] += vertical_offset
        
        return foot