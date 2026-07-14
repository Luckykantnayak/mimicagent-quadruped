from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_BUTTERFLY_KICK_GLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Butterfly Kick Glide: Dolphin-like undulating motion.
    
    - All four legs sweep synchronously (no phase offset)
    - Base executes sinusoidal pitch oscillation coordinated with leg motion
    - Forward propulsion through body wave coordination
    - Compression phase [0.0-0.5]: legs sweep backward, base pitches down
    - Recovery phase [0.5-1.0]: legs sweep forward, base pitches up
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Undulation frequency (Hz)
        
        # Leg motion parameters - balanced co-optimization to eliminate penetration
        self.sweep_length = 0.15  # Fore-aft sweep amplitude in body frame
        self.sweep_height_front = 0.05  # Vertical lift during recovery (front legs)
        self.sweep_height_rear = 0.015  # Minimal vertical lift during recovery (rear legs)
        self.stance_depth_front = 0.030  # Moderate downward extension during power stroke (front legs)
        self.stance_depth_rear = 0.015  # Reduced downward extension for rear legs (balanced with offset)
        
        # Base foot positions (BODY frame) with calibrated offset for rear legs
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            pos = v.copy()
            # Balanced vertical offset for rear legs: covers depression with 20% safety margin
            if k.startswith('R'):
                pos[2] += 0.018  # Offset = 0.018m, Depression = 0.015m, Ratio = 1.2
            self.base_feet_pos_body[k] = pos
        
        # All legs synchronized (no phase offset)
        self.phase_offsets = {
            leg_names[0]: 0.0,
            leg_names[1]: 0.0,
            leg_names[2]: 0.0,
            leg_names[3]: 0.0,
        }
        
        # Base motion parameters - tuned to maintain contact while enabling undulation
        self.vx_forward = 0.8  # Forward velocity magnitude
        self.vz_amplitude = 0.08  # Reduced vertical oscillation amplitude (prevents flight)
        self.pitch_rate_amplitude = 1.0  # Moderate pitch angular velocity amplitude (rad/s)
        self.max_pitch_angle = 0.26  # Maximum pitch deviation (~15 degrees)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Command storage
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base pose with undulating pitch and forward motion.
        
        Phase structure:
        [0.0-0.3]: Compression - pitch down, descend slightly
        [0.3-0.5]: Max extension - hold pitch, level flight
        [0.5-0.8]: Recovery - pitch up, ascend slightly
        [0.8-1.0]: Neutral prep - level out, prepare for next cycle
        """
        
        # Forward velocity (constant throughout)
        vx = self.vx_forward
        vy = 0.0
        
        # Vertical velocity (sinusoidal undulation) - reduced amplitude
        vz = self.vz_amplitude * np.sin(2 * np.pi * phase - np.pi / 2)
        
        # Pitch rate (smooth sinusoidal with phase-specific shaping)
        if phase < 0.3:
            # Compression: negative pitch rate (nose down)
            pitch_rate = -self.pitch_rate_amplitude * np.sin(np.pi * phase / 0.3)
        elif phase < 0.5:
            # Max extension: hold pitch (zero rate)
            pitch_rate = 0.0
        elif phase < 0.8:
            # Recovery: positive pitch rate (nose up)
            progress = (phase - 0.5) / 0.3
            pitch_rate = self.pitch_rate_amplitude * np.sin(np.pi * progress)
        else:
            # Neutral prep: small negative rate to level out
            progress = (phase - 0.8) / 0.2
            pitch_rate = -0.5 * self.pitch_rate_amplitude * np.sin(np.pi * progress)
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
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
        Compute synchronized leg trajectories in body frame.
        
        All legs move in phase:
        [0.0-0.5]: Backward and downward sweep (power stroke)
        [0.5-1.0]: Forward and upward sweep (recovery stroke)
        
        Front legs: moderate depression during power stroke, moderate lift during recovery
        Rear legs: reduced depression (calibrated to avoid penetration), minimal lift to maintain stability
        """
        
        # Get base position for this leg
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if front or rear leg
        is_front = leg_name.startswith('F')
        sweep_height = self.sweep_height_front if is_front else self.sweep_height_rear
        stance_depth = self.stance_depth_front if is_front else self.stance_depth_rear
        
        # Synchronized phase (all legs move together)
        leg_phase = phase
        
        if leg_phase < 0.5:
            # Power stroke: backward and downward sweep [0.0-0.5]
            progress = leg_phase / 0.5  # 0 -> 1 over first half
            
            # Fore-aft: sweep from forward to backward
            foot[0] += self.sweep_length * (0.5 - progress)
            
            # Vertical: press downward during compression with smooth sinusoidal curve
            # Maximum depression at phase 0.25 (mid-power stroke)
            depression_curve = np.sin(np.pi * progress)
            foot[2] -= stance_depth * depression_curve
            
        else:
            # Recovery stroke: forward and upward sweep [0.5-1.0]
            progress = (leg_phase - 0.5) / 0.5  # 0 -> 1 over second half
            
            # Fore-aft: sweep from backward to forward
            foot[0] += self.sweep_length * (progress - 0.5)
            
            # Vertical: gentle lift during recovery (front legs lift more, rear legs minimal)
            lift_curve = np.sin(np.pi * progress)
            foot[2] += sweep_height * lift_curve
        
        return foot