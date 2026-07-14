from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_BUTTERFLY_KICK_GLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Butterfly kick / dolphin glide motion generator.
    
    All four legs sweep synchronously backward-then-forward in a wave pattern
    while the base pitches rhythmically, creating forward gliding propulsion
    through coordinated momentum transfer.
    
    Phase structure:
      [0.0, 0.3]: Power stroke - pitch down, legs sweep back/down
      [0.3, 0.5]: Extension transition - pitch reverses, legs hold extended
      [0.5, 0.8]: Recovery sweep - pitch up, legs sweep forward/up
      [0.8, 1.0]: Reset - level out, legs return to neutral
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz) - slower for smooth undulation
        
        # Base foot positions in BODY frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Leg sweep parameters
        self.sweep_amplitude = 0.15  # Forward-backward range in body frame (m)
        self.vertical_amplitude = 0.05  # Up-down range during sweep (m)
        
        # Base motion parameters
        self.forward_velocity = 0.5  # Average forward velocity (m/s)
        self.pitch_amplitude = 0.35  # Pitch oscillation amplitude (rad) ~20 degrees
        self.pitch_freq_multiplier = 1.0  # Pitch frequency relative to gait cycle
        
        # Vertical velocity amplitude from pitch (m/s)
        self.vz_amplitude = 0.15
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # All legs move synchronously - no phase offsets
        self.phase_offsets = {leg: 0.0 for leg in leg_names}

    def update_base_motion(self, phase, dt):
        """
        Update base pose with sinusoidal pitch oscillation and forward glide.
        
        Phase-dependent behavior:
          [0.0, 0.3]: Pitch down (negative rate), forward + slightly down
          [0.3, 0.5]: Pitch reverses to up (positive rate), forward + transition to up
          [0.5, 0.8]: Pitch up (positive rate), forward + upward
          [0.8, 1.0]: Pitch levels (zero rate), forward + level
        """
        
        # Forward velocity - constant glide with slight modulation during power stroke
        if phase < 0.3:
            # Boost during power stroke
            vx = self.forward_velocity * (1.0 + 0.3 * (phase / 0.3))
        else:
            # Maintain momentum
            vx = self.forward_velocity * 1.3 * (1.0 - 0.3 * (phase - 0.3) / 0.7)
        
        # Vertical velocity follows pitch motion
        if phase < 0.3:
            # Downward during pitch down
            vz = -self.vz_amplitude * np.sin(np.pi * phase / 0.3)
        elif phase < 0.5:
            # Transition from down to up
            local_phase = (phase - 0.3) / 0.2
            vz = self.vz_amplitude * np.sin(np.pi * local_phase)
        elif phase < 0.8:
            # Upward during pitch up
            local_phase = (phase - 0.5) / 0.3
            vz = self.vz_amplitude * (1.0 - local_phase)
        else:
            # Level out
            local_phase = (phase - 0.8) / 0.2
            vz = self.vz_amplitude * 0.1 * (1.0 - local_phase)
        
        # Pitch rate - sinusoidal undulation
        # Negative rate = pitch down (nose down)
        # Positive rate = pitch up (nose up)
        pitch_phase = 2.0 * np.pi * self.pitch_freq_multiplier * phase
        
        if phase < 0.3:
            # Power stroke: pitch down
            pitch_rate = -2.0 * np.pi * self.pitch_amplitude * self.freq * np.cos(pitch_phase)
        elif phase < 0.5:
            # Transition: pitch reverses to up
            pitch_rate = 2.0 * np.pi * self.pitch_amplitude * self.freq * np.cos(pitch_phase)
        elif phase < 0.8:
            # Recovery: continue pitch up
            pitch_rate = 2.0 * np.pi * self.pitch_amplitude * self.freq * np.cos(pitch_phase)
        else:
            # Reset: damp pitch rate to zero
            local_phase = (phase - 0.8) / 0.2
            pitch_rate = 2.0 * np.pi * self.pitch_amplitude * self.freq * np.cos(pitch_phase) * (1.0 - local_phase)
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, 0.0, vz])
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
        Compute synchronized leg sweep trajectory in body frame.
        
        All legs perform identical motion:
          [0.0, 0.3]: Sweep backward and downward (power stroke)
          [0.3, 0.5]: Hold at maximum backward extension
          [0.5, 0.8]: Sweep forward and upward (recovery)
          [0.8, 1.0]: Return to neutral position (reset)
        """
        
        # Start from base position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Synchronized leg phase (no offset between legs)
        leg_phase = phase
        
        # Compute forward-backward sweep position
        if leg_phase < 0.3:
            # Power stroke: sweep backward
            # Progress from 0 (neutral) to 1 (max backward)
            progress = leg_phase / 0.3
            x_offset = self.sweep_amplitude * progress
            z_offset = -self.vertical_amplitude * progress
            
        elif leg_phase < 0.5:
            # Extension hold: maintain max backward position
            x_offset = self.sweep_amplitude
            z_offset = -self.vertical_amplitude
            
        elif leg_phase < 0.8:
            # Recovery sweep: sweep forward and upward
            # Progress from 0 (max backward) to 1 (max forward)
            progress = (leg_phase - 0.5) / 0.3
            x_offset = self.sweep_amplitude * (1.0 - 2.0 * progress)
            # Smooth upward arc using sine
            z_offset = -self.vertical_amplitude * (1.0 - progress) + self.vertical_amplitude * 0.3 * np.sin(np.pi * progress)
            
        else:
            # Reset: return to neutral
            # Progress from 0 (forward) to 1 (neutral)
            progress = (leg_phase - 0.8) / 0.2
            x_offset = -self.sweep_amplitude * (1.0 - progress)
            z_offset = self.vertical_amplitude * 0.3 * (1.0 - progress)
        
        # Apply offsets in body frame
        # x: forward-backward sweep
        # z: vertical component of sweep
        foot[0] -= x_offset  # Negative because backward is negative x in body frame
        foot[2] += z_offset
        
        return foot