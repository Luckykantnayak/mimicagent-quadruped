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
        
        # Leg motion parameters
        self.sweep_length = 0.15  # Fore-aft sweep amplitude in body frame
        self.sweep_height_front = 0.06  # Vertical lift during recovery (front legs)
        self.sweep_height_rear = 0.03  # Vertical lift during recovery (rear legs, less than front)
        self.stance_depth = 0.04  # Downward extension during power stroke
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # All legs synchronized (no phase offset)
        self.phase_offsets = {
            leg_names[0]: 0.0,
            leg_names[1]: 0.0,
            leg_names[2]: 0.0,
            leg_names[3]: 0.0,
        }
        
        # Base motion parameters
        self.vx_forward = 0.8  # Forward velocity magnitude
        self.vz_amplitude = 0.12  # Vertical oscillation amplitude
        self.pitch_rate_amplitude = 1.2  # Pitch angular velocity amplitude (rad/s)
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
        
        # Vertical velocity (sinusoidal undulation)
        # Negative in [0.0-0.3] (descend), positive in [0.5-0.8] (ascend)
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
        
        Front legs lift more during recovery; rear legs stay lower for stability.
        """
        
        # Get base position for this leg
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if front or rear leg
        is_front = leg_name.startswith('F')
        sweep_height = self.sweep_height_front if is_front else self.sweep_height_rear
        
        # Synchronized phase (all legs move together)
        leg_phase = phase
        
        if leg_phase < 0.5:
            # Power stroke: backward and downward sweep [0.0-0.5]
            progress = leg_phase / 0.5  # 0 -> 1 over first half
            
            # Fore-aft: sweep from forward to backward
            foot[0] += self.sweep_length * (0.5 - progress)
            
            # Vertical: press downward during compression
            # Maximum depression at phase 0.3-0.4
            depression_curve = np.sin(np.pi * progress)
            foot[2] -= self.stance_depth * depression_curve
            
        else:
            # Recovery stroke: forward and upward sweep [0.5-1.0]
            progress = (leg_phase - 0.5) / 0.5  # 0 -> 1 over second half
            
            # Fore-aft: sweep from backward to forward
            foot[0] += self.sweep_length * (progress - 0.5)
            
            # Vertical: lift upward during recovery
            # Arc trajectory with peak at mid-recovery
            lift_curve = np.sin(np.pi * progress)
            foot[2] += sweep_height * lift_curve
            
            # Rear legs: reduced lift, more contact retention
            if not is_front:
                # Scale down vertical motion for rear legs
                foot[2] -= sweep_height * 0.3 * lift_curve
        
        return foot