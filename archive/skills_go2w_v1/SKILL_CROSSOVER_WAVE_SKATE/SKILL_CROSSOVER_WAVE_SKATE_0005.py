from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CROSSOVER_WAVE_SKATE_MotionGenerator(BaseMotionGenerator):
    """
    Crossover Wave Skate: A skating-style locomotion where the robot glides forward
    while legs execute synchronized inward-outward crossover sweeps forming a traveling
    wave from front to rear. The base exhibits smooth roll and yaw oscillations
    synchronized with the leg wave, creating fluid, rhythmic skating motion.
    
    Motion characteristics:
    - Forward glide with lateral carving sweeps
    - Traveling wave propagates front-to-rear with 0.25 phase delay
    - Left-right sides offset by 0.5 phase
    - Base roll and yaw coordinate with leg motion for weight shift
    - All four wheels maintain contact (with partial unloading during crossover)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Skating rhythm frequency (Hz)
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Leg motion parameters
        self.lateral_sweep_amplitude = 0.12  # Lateral sweep distance (m)
        self.crossover_depth = 0.08  # How far legs cross past centerline (m)
        self.vertical_unload = 0.015  # Slight vertical lift during crossover (m)
        self.forward_component = 0.03  # Forward reach during outward carve (m)
        
        # Phase offsets for traveling wave
        # Left side: FL carves at 0.0, RL carves at 0.25 (front-to-rear delay)
        # Right side: FR carves at 0.5, RR carves at 0.75 (half cycle offset + front-to-rear)
        self.phase_offsets = {
            leg_names[0] if leg_names[0].startswith('FL') else [l for l in leg_names if l.startswith('FL')][0]: 0.0,
            leg_names[1] if leg_names[1].startswith('FR') else [l for l in leg_names if l.startswith('FR')][0]: 0.5,
            leg_names[2] if leg_names[2].startswith('RL') else [l for l in leg_names if l.startswith('RL')][0]: 0.25,
            leg_names[3] if leg_names[3].startswith('RR') else [l for l in leg_names if l.startswith('RR')][0]: 0.75,
        }
        
        # Normalize leg name mapping
        self.leg_map = {}
        for leg in leg_names:
            if leg.startswith('FL'):
                self.leg_map['FL'] = leg
            elif leg.startswith('FR'):
                self.leg_map['FR'] = leg
            elif leg.startswith('RL'):
                self.leg_map['RL'] = leg
            elif leg.startswith('RR'):
                self.leg_map['RR'] = leg
        
        # Base velocity parameters
        self.vx_forward = 0.5  # Steady forward skating velocity (m/s)
        self.vy_amplitude = 0.08  # Lateral velocity oscillation amplitude (m/s)
        
        # Base angular velocity parameters
        self.roll_rate_amplitude = 0.35  # Roll rate amplitude (rad/s) -> ~12 deg integrated
        self.yaw_rate_amplitude = 0.25  # Yaw rate amplitude (rad/s) -> ~8 deg integrated
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands to create skating motion with coordinated
        roll and yaw oscillations.
        
        Phase coordination:
        - phase 0.0-0.5: left side carving (roll left, yaw right to counter)
        - phase 0.5-1.0: right side carving (roll right, yaw left to counter)
        """
        
        # Forward velocity: steady glide
        vx = self.vx_forward
        
        # Lateral velocity: oscillates left-right with carving rhythm
        # Negative (left) during left carve (phase 0-0.5), positive (right) during right carve
        vy = self.vy_amplitude * np.sin(2 * np.pi * phase)
        
        # Roll rate: negative (left) for phase 0-0.5, positive (right) for phase 0.5-1.0
        # Creates weight shift toward carving side
        roll_rate = -self.roll_rate_amplitude * np.cos(2 * np.pi * phase)
        
        # Yaw rate: positive (right) during left carve to counter lateral drift,
        # negative (left) during right carve
        # Phase shifted to counter the carving motion
        yaw_rate = self.yaw_rate_amplitude * np.sin(2 * np.pi * phase)
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, yaw_rate])
        
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
        Compute foot position for skating crossover wave motion.
        
        Each leg cycles through:
        1. Outward carve: extend laterally away from centerline (firm contact)
        2. Hold extended: maintain position as wave propagates
        3. Inward crossover: sweep inward past centerline (partial unload)
        4. Return to neutral: prepare for next carve
        
        Traveling wave: front legs lead by 0.25 phase relative to rear legs
        Left-right offset: 0.5 phase between sides
        """
        
        # Get leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg side for lateral direction
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            lateral_sign = -1.0  # Left side: negative y is outward (left)
        else:
            lateral_sign = 1.0   # Right side: positive y is outward (right)
        
        # Determine front/rear for forward component
        if leg_name.startswith('FL') or leg_name.startswith('FR'):
            is_front = True
        else:
            is_front = False
        
        # Carving phase pattern (smooth sinusoidal wave)
        # leg_phase 0.0: start outward carve
        # leg_phase 0.25: maximum extension
        # leg_phase 0.5: start inward crossover
        # leg_phase 0.75: maximum crossover (crossed under body)
        # leg_phase 1.0: return to neutral
        
        # Lateral displacement: sinusoidal from outward extension to inward crossover
        # -1.0 at leg_phase=0.25 (max outward), +1.0 at leg_phase=0.75 (max inward)
        lateral_cycle = np.cos(2 * np.pi * leg_phase)
        
        # Map to actual lateral displacement
        # Outward carve: extend by lateral_sweep_amplitude
        # Inward crossover: cross by crossover_depth
        if lateral_cycle < 0:
            # Carving phase (leg_phase ~0.0-0.5): extend outward
            lateral_displacement = lateral_sign * self.lateral_sweep_amplitude * (-lateral_cycle)
            vertical_offset = 0.0  # Firm contact during carve
            forward_offset = self.forward_component * (-lateral_cycle) if is_front else 0.0
        else:
            # Crossover phase (leg_phase ~0.5-1.0): cross inward
            lateral_displacement = -lateral_sign * self.crossover_depth * lateral_cycle
            # Slight vertical lift to reduce contact during crossover
            vertical_offset = self.vertical_unload * np.sin(np.pi * lateral_cycle)
            forward_offset = 0.0
        
        # Apply offsets
        foot[0] += forward_offset  # Forward component during carve
        foot[1] += lateral_displacement  # Lateral sweep
        foot[2] += vertical_offset  # Vertical unload during crossover
        
        return foot