from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_WINDSHIELD_WIPER_YAW_MotionGenerator(BaseMotionGenerator):
    """
    Windshield-wiper yaw oscillation: slow 70° clockwise sweep, rapid counterclockwise return, pause.
    
    Phase structure:
    - [0.0, 0.6]: Slow clockwise yaw rotation (~70°)
    - [0.6, 0.7]: Rapid counterclockwise return (~70°)
    - [0.7, 1.0]: Pause and stabilization
    
    All four feet remain in contact throughout, stepping in small coordinated arcs
    to enable in-place yaw rotation without net translation.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for deliberate oscillation
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Yaw motion parameters
        # Slow sweep: 70° over 0.6 of cycle → yaw_rate = 70° / (0.6 / freq)
        # Target: ~70 degrees = ~1.22 radians
        self.target_angle = np.deg2rad(70.0)
        self.slow_phase_duration = 0.6
        self.fast_phase_duration = 0.1
        
        # Yaw rates (rad/s)
        # During slow sweep: integrate over (slow_phase_duration / freq) seconds
        # yaw_rate * (slow_phase_duration / freq) = target_angle
        self.yaw_rate_slow = self.target_angle / (self.slow_phase_duration / self.freq)
        
        # During fast return: unwind same angle in shorter time
        # yaw_rate * (fast_phase_duration / freq) = -target_angle
        self.yaw_rate_fast = -self.target_angle / (self.fast_phase_duration / self.freq)
        
        # Foot stepping parameters for in-place rotation
        # Diagonal pairs alternate stepping to enable yaw without translation
        self.step_height = 0.03  # Low clearance for stability
        self.arc_radius = 0.05   # Small lateral adjustment per step
        
        # Phase offsets for diagonal coordination
        # FL+RR step together (group 1), FR+RL step together (group 2)
        # Offset by 0.5 of the internal stepping cycle
        self.phase_offsets = {
            leg_names[0]: 0.0,   # FL
            leg_names[1]: 0.5,   # FR
            leg_names[2]: 0.5,   # RL
            leg_names[3]: 0.0,   # RR
        }
        
        # Internal stepping frequency during slow sweep
        # Multiple steps per yaw sweep for continuous contact
        self.step_freq_multiplier = 4.0  # 4 stepping cycles during slow sweep
        
        # Duty cycle for stepping (high to maintain contact)
        self.duty_cycle = 0.7

    def update_base_motion(self, phase, dt):
        """
        Update base yaw according to phase:
        - [0.0, 0.6]: Positive yaw rate (clockwise sweep)
        - [0.6, 0.7]: Negative yaw rate (counterclockwise return)
        - [0.7, 1.0]: Zero yaw rate (pause)
        
        No linear translation (in-place rotation).
        """
        if phase < 0.6:
            # Slow clockwise sweep
            yaw_rate = self.yaw_rate_slow
        elif phase < 0.7:
            # Rapid counterclockwise return
            yaw_rate = self.yaw_rate_fast
        else:
            # Pause phase
            yaw_rate = 0.0
        
        # Set velocities (no linear motion, only yaw)
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

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with small stepping arcs to enable yaw rotation.
        
        During slow sweep and rapid return: diagonal pairs alternate small steps.
        During pause: all feet in stance.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine which leg group
        is_fl = leg_name.startswith('FL')
        is_rr = leg_name.startswith('RR')
        is_fr = leg_name.startswith('FR')
        is_rl = leg_name.startswith('RL')
        
        # Group 1: FL, RR (phase offset 0.0)
        # Group 2: FR, RL (phase offset 0.5)
        if is_fl or is_rr:
            leg_phase_offset = self.phase_offsets[self.leg_names[0]]
        else:
            leg_phase_offset = self.phase_offsets[self.leg_names[1]]
        
        if phase < 0.6:
            # Slow sweep phase: continuous stepping
            # Map phase [0.0, 0.6] to multiple step cycles
            internal_phase = (phase / 0.6) * self.step_freq_multiplier
            internal_phase = internal_phase % 1.0
            leg_phase = (internal_phase + leg_phase_offset) % 1.0
            
            # Apply small stepping motion
            foot = self._apply_step_motion(foot, leg_phase, leg_name, scale=1.0)
            
        elif phase < 0.7:
            # Rapid return phase: quick repositioning
            # Map phase [0.6, 0.7] to single rapid step cycle
            internal_phase = (phase - 0.6) / 0.1
            leg_phase = (internal_phase + leg_phase_offset) % 1.0
            
            # Apply rapid stepping motion (higher priority, faster)
            foot = self._apply_step_motion(foot, leg_phase, leg_name, scale=1.5)
            
        else:
            # Pause phase: all feet in stance, no motion
            pass
        
        return foot

    def _apply_step_motion(self, foot, leg_phase, leg_name, scale=1.0):
        """
        Apply small stepping motion to enable in-place yaw.
        
        Stance phase: foot stationary in body frame
        Swing phase: small arc trajectory (lateral and vertical adjustment)
        
        Scale allows adjustment of step magnitude (e.g., faster steps during rapid return)
        """
        if leg_phase < self.duty_cycle:
            # Stance phase: foot remains at base position
            return foot
        else:
            # Swing phase: small arc motion
            swing_progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)
            
            # Vertical: low arc
            arc_angle = np.pi * swing_progress
            foot[2] += self.step_height * scale * np.sin(arc_angle)
            
            # Lateral adjustment for yaw rotation
            # Feet move in small circular arcs in body frame to facilitate rotation
            # Front legs move slightly forward-lateral, rear legs move slightly back-lateral
            is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
            is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
            
            # Lateral oscillation
            lateral_offset = self.arc_radius * scale * np.sin(2 * np.pi * swing_progress)
            if is_left:
                foot[1] += lateral_offset
            else:
                foot[1] -= lateral_offset
            
            # Longitudinal adjustment (smaller)
            longitudinal_offset = 0.5 * self.arc_radius * scale * np.cos(2 * np.pi * swing_progress)
            if is_front:
                foot[0] += longitudinal_offset
            else:
                foot[0] -= longitudinal_offset
        
        return foot