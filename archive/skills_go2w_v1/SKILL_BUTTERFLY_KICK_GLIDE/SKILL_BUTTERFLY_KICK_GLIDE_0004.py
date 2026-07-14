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
        self.vertical_amplitude = 0.04  # Up-down range during sweep (m)
        
        # Base motion parameters
        self.forward_velocity = 0.5  # Average forward velocity (m/s)
        self.pitch_amplitude = 0.25  # Pitch oscillation amplitude (rad)
        self.pitch_freq_multiplier = 1.0  # Pitch frequency relative to gait cycle
        
        # Vertical velocity amplitude from pitch (m/s)
        self.vz_amplitude = 0.035
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Phase offsets for wave propagation
        self.phase_offsets = {}
        for i, leg in enumerate(leg_names):
            if i < 2:  # Front legs
                self.phase_offsets[leg] = 0.0
            else:  # Rear legs
                self.phase_offsets[leg] = 0.28
        
        # Identify front vs rear legs based on initial x position
        self.is_front_leg = {}
        self.leg_longitudinal_dist = {}
        for leg in leg_names:
            x_pos = self.base_feet_pos_body[leg][0]
            self.is_front_leg[leg] = x_pos > 0.0
            self.leg_longitudinal_dist[leg] = abs(x_pos)

    def update_base_motion(self, phase, dt):
        """
        Update base pose with sinusoidal pitch oscillation and forward glide.
        """
        
        # Forward velocity - constant glide with slight modulation during power stroke
        if phase < 0.3:
            vx = self.forward_velocity * (1.0 + 0.2 * (phase / 0.3))
        else:
            vx = self.forward_velocity * (1.2 - 0.2 * (phase - 0.3) / 0.7)
        
        # Vertical velocity follows pitch motion
        if phase < 0.3:
            vz = -self.vz_amplitude * 0.5 * np.sin(np.pi * phase / 0.3)
        elif phase < 0.55:
            local_phase = (phase - 0.3) / 0.25
            vz = self.vz_amplitude * 0.8 * np.sin(np.pi * local_phase)
        else:
            vz = 0.0
        
        # Pitch rate - sinusoidal undulation with smooth envelope
        pitch_phase = 2.0 * np.pi * self.pitch_freq_multiplier * phase
        pitch_rate_base = 2.0 * np.pi * self.pitch_amplitude * self.freq
        
        if phase < 0.3:
            envelope = np.sin(np.pi * phase / 0.3)
            pitch_rate = -pitch_rate_base * np.cos(pitch_phase) * envelope
        elif phase < 0.5:
            pitch_rate = pitch_rate_base * np.cos(pitch_phase)
        elif phase < 0.8:
            local_phase = (phase - 0.5) / 0.3
            envelope = 1.0 - 0.3 * local_phase
            pitch_rate = pitch_rate_base * np.cos(pitch_phase) * envelope
        else:
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
        Compute synchronized leg sweep trajectory in body frame with separated
        intrinsic motion and pitch compensation.
        """
        
        # Start from base position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Apply phase offset for this leg
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Determine if this is a front or rear leg
        is_front = self.is_front_leg[leg_name]
        longitudinal_dist = self.leg_longitudinal_dist[leg_name]
        
        # Compute current pitch angle for geometric compensation
        pitch_phase_angle = 2.0 * np.pi * self.pitch_freq_multiplier * phase
        current_pitch = self.pitch_amplitude * np.sin(pitch_phase_angle)
        
        # Calculate pitch-induced vertical displacement at this leg's position
        # This is the geometric correction needed independent of leg phase
        if is_front:
            pitch_z_geometric = longitudinal_dist * np.sin(current_pitch)
        else:
            pitch_z_geometric = -longitudinal_dist * np.sin(current_pitch)
        
        # Compute phase-independent pitch compensation with safety margin
        # Rear legs need strong upward offset when pitch is negative (nose down)
        # Front legs need upward offset when pitch is positive (nose up)
        if is_front:
            pitch_compensation = max(0.0, pitch_z_geometric) * 1.5
        else:
            # Rear legs get amplified compensation with safety margin
            pitch_compensation = max(0.0, -pitch_z_geometric) * 1.8
        
        # Add static ground clearance offset for rear legs to handle worst-case geometry
        if not is_front:
            # Additional constant offset to ensure clearance during maximum pitch-down
            # Maximum pitch is 0.25 rad, longitudinal_dist ~0.3m, sin(0.25) ~0.247
            # Worst case drop: 0.3 * 0.247 = 0.074m, with margin: 0.074 * 1.5 = 0.111m
            static_clearance_offset = longitudinal_dist * np.sin(self.pitch_amplitude) * 0.5
        else:
            static_clearance_offset = 0.0
        
        # Compute intrinsic leg sweep motion (aesthetic dolphin kick)
        if leg_phase < 0.3:
            # Power stroke: sweep backward with minimal downward motion
            progress = leg_phase / 0.3
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            x_offset = self.sweep_amplitude * smooth_progress
            
            # Intrinsic vertical motion during power stroke - minimal for both legs
            if is_front:
                z_intrinsic = -self.vertical_amplitude * 0.3 * smooth_progress
            else:
                # Rear legs: no downward intrinsic motion during power stroke
                z_intrinsic = 0.0
            
        elif leg_phase < 0.5:
            # Extension hold: maintain max backward position
            x_offset = self.sweep_amplitude
            
            # Minimal intrinsic vertical offset during hold
            if is_front:
                z_intrinsic = -self.vertical_amplitude * 0.2
            else:
                z_intrinsic = 0.0
            
        elif leg_phase < 0.8:
            # Recovery sweep: sweep forward with controlled upward arc
            progress = (leg_phase - 0.5) / 0.3
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            x_offset = self.sweep_amplitude * (1.0 - 2.0 * smooth_progress)
            
            # Minimal intrinsic arc during recovery
            if is_front:
                arc_height = self.vertical_amplitude * 0.08 * np.sin(np.pi * smooth_progress)
                z_intrinsic = -self.vertical_amplitude * 0.2 * (1.0 - progress) + arc_height
            else:
                # Rear legs: minimal arc, rely on compensation for clearance
                arc_height = self.vertical_amplitude * 0.05 * np.sin(np.pi * smooth_progress)
                z_intrinsic = arc_height
            
        else:
            # Reset: return to neutral smoothly
            progress = (leg_phase - 0.8) / 0.2
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            x_offset = -self.sweep_amplitude * (1.0 - smooth_progress)
            
            # Smooth return to neutral
            if is_front:
                z_intrinsic = self.vertical_amplitude * 0.15 * (1.0 - smooth_progress)
            else:
                z_intrinsic = 0.0
        
        # Combine intrinsic motion with pitch compensation and static offset
        z_offset = z_intrinsic + pitch_compensation + static_clearance_offset
        
        # Apply offsets in body frame
        foot[0] -= x_offset  # Negative because backward is negative x in body frame
        foot[2] += z_offset
        
        return foot