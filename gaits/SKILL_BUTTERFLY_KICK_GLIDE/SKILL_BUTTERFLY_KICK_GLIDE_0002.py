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
        self.vertical_amplitude = 0.04  # Up-down range during sweep (m) - reduced to prevent penetration
        
        # Base motion parameters
        self.forward_velocity = 0.5  # Average forward velocity (m/s)
        self.pitch_amplitude = 0.25  # Pitch oscillation amplitude (rad) - reduced from 0.35
        self.pitch_freq_multiplier = 1.0  # Pitch frequency relative to gait cycle
        
        # Vertical velocity amplitude from pitch (m/s)
        self.vz_amplitude = 0.08  # Reduced from 0.15 to minimize airtime
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Introduce slight phase offsets to prevent simultaneous liftoff
        # Front legs lead, rear legs lag slightly
        self.phase_offsets = {}
        for i, leg in enumerate(leg_names):
            if i < 2:  # Front legs (typically indices 0, 1)
                self.phase_offsets[leg] = 0.0
            else:  # Rear legs (typically indices 2, 3)
                self.phase_offsets[leg] = 0.15  # 15% phase lag
        
        # Identify front vs rear legs based on initial x position
        self.is_front_leg = {}
        for leg in leg_names:
            self.is_front_leg[leg] = self.base_feet_pos_body[leg][0] > 0.0

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
            vx = self.forward_velocity * (1.0 + 0.2 * (phase / 0.3))
        else:
            # Maintain momentum with smooth decay
            vx = self.forward_velocity * (1.2 - 0.2 * (phase - 0.3) / 0.7)
        
        # Vertical velocity follows pitch motion - reduced amplitude and duration
        if phase < 0.3:
            # Slight downward during pitch down
            vz = -self.vz_amplitude * 0.5 * np.sin(np.pi * phase / 0.3)
        elif phase < 0.5:
            # Brief transition upward
            local_phase = (phase - 0.3) / 0.2
            vz = self.vz_amplitude * 0.6 * np.sin(np.pi * local_phase)
        elif phase < 0.65:
            # Short upward pulse during early recovery
            local_phase = (phase - 0.5) / 0.15
            vz = self.vz_amplitude * 0.4 * (1.0 - local_phase)
        else:
            # Return to neutral quickly to maintain ground proximity
            vz = 0.0
        
        # Pitch rate - sinusoidal undulation with smooth envelope
        pitch_phase = 2.0 * np.pi * self.pitch_freq_multiplier * phase
        pitch_rate_base = 2.0 * np.pi * self.pitch_amplitude * self.freq
        
        if phase < 0.3:
            # Power stroke: pitch down with smooth acceleration
            envelope = np.sin(np.pi * phase / 0.3)
            pitch_rate = -pitch_rate_base * np.cos(pitch_phase) * envelope
        elif phase < 0.5:
            # Transition: pitch reverses to up
            pitch_rate = pitch_rate_base * np.cos(pitch_phase)
        elif phase < 0.8:
            # Recovery: continue pitch up with damping
            local_phase = (phase - 0.5) / 0.3
            envelope = 1.0 - 0.3 * local_phase
            pitch_rate = pitch_rate_base * np.cos(pitch_phase) * envelope
        else:
            # Reset: damp pitch rate to zero smoothly
            local_phase = (phase - 0.8) / 0.2
            envelope = (1.0 - local_phase) ** 2
            pitch_rate = pitch_rate_base * np.cos(pitch_phase) * envelope * 0.5
        
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
        
        All legs perform similar motion with phase offsets and position-dependent compensation:
          [0.0, 0.3]: Sweep backward and downward (power stroke)
          [0.3, 0.5]: Hold at maximum backward extension
          [0.5, 0.8]: Sweep forward and upward (recovery)
          [0.8, 1.0]: Return to neutral position (reset)
        """
        
        # Start from base position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Apply phase offset for this leg
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Determine if this is a front or rear leg
        is_front = self.is_front_leg[leg_name]
        
        # Adjust vertical amplitude based on leg position to prevent ground penetration
        # Rear legs use reduced vertical motion and inverted offset during pitch down
        if is_front:
            vert_scale = 1.0
            z_compensation = 0.0
        else:
            vert_scale = 0.6  # Reduce vertical excursion for rear legs
            # Add upward compensation when base pitches down
            if leg_phase < 0.3:
                z_compensation = 0.02 * (leg_phase / 0.3)
            else:
                z_compensation = 0.02 * max(0.0, 1.0 - (leg_phase - 0.3) / 0.2)
        
        # Compute forward-backward sweep position
        if leg_phase < 0.3:
            # Power stroke: sweep backward with minimal downward motion
            progress = leg_phase / 0.3
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)  # Smooth step
            x_offset = self.sweep_amplitude * smooth_progress
            # Rear legs have reduced downward motion
            z_offset = -self.vertical_amplitude * vert_scale * smooth_progress * 0.5 + z_compensation
            
        elif leg_phase < 0.5:
            # Extension hold: maintain max backward position with ground contact
            x_offset = self.sweep_amplitude
            # Minimal vertical offset, biased upward for rear legs
            z_offset = -self.vertical_amplitude * vert_scale * 0.3 + z_compensation
            
        elif leg_phase < 0.8:
            # Recovery sweep: sweep forward with controlled upward motion
            progress = (leg_phase - 0.5) / 0.3
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            x_offset = self.sweep_amplitude * (1.0 - 2.0 * smooth_progress)
            # Reduced upward arc with smooth trajectory
            arc_height = self.vertical_amplitude * vert_scale * 0.4 * np.sin(np.pi * smooth_progress)
            z_offset = -self.vertical_amplitude * vert_scale * 0.3 * (1.0 - progress) + arc_height
            
        else:
            # Reset: return to neutral smoothly
            progress = (leg_phase - 0.8) / 0.2
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            x_offset = -self.sweep_amplitude * (1.0 - smooth_progress)
            z_offset = self.vertical_amplitude * vert_scale * 0.2 * (1.0 - smooth_progress)
        
        # Apply offsets in body frame
        foot[0] -= x_offset  # Negative because backward is negative x in body frame
        foot[2] += z_offset
        
        return foot