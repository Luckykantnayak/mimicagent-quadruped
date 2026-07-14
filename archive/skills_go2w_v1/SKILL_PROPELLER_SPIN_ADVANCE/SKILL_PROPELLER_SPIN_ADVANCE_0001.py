from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PROPELLER_SPIN_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Propeller spin advance gait with continuous circular leg rotation.
    
    - All four legs rotate continuously in synchronized vertical circles
    - Front legs (FL, FR) and rear legs (RL, RR) are 180 degrees out of phase
    - Base maintains forward velocity with gentle pitch oscillations
    - Contact occurs briefly when legs pass through bottom of rotation arc
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Rotation frequency (cycles per second)
        
        # Circular rotation parameters
        self.circle_radius = 0.12  # Radius of vertical circular leg rotation
        self.circle_center_offset_x = 0.0  # Forward offset of circle center from base foot position
        self.circle_center_offset_z = -0.05  # Vertical offset of circle center
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets: front legs synchronized, rear legs 180 degrees out of phase
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('F'):  # Front legs: FL, FR
                self.phase_offsets[leg] = 0.0
            else:  # Rear legs: RL, RR
                self.phase_offsets[leg] = 0.5
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters
        self.vx_forward = 0.6  # Sustained forward velocity
        self.pitch_amplitude = 0.15  # Amplitude of pitch oscillation (radians)
        self.pitch_freq_multiplier = 2.0  # Pitch oscillates at 2x the leg rotation frequency

    def update_base_motion(self, phase, dt):
        """
        Update base with sustained forward velocity and sinusoidal pitch oscillation.
        
        Pitch oscillation coordinates body orientation with leg rotation phases:
        - phase 0.0-0.25: pitch down (rear legs at bottom)
        - phase 0.25-0.5: pitch up (leveling)
        - phase 0.5-0.75: pitch up (front legs at bottom)
        - phase 0.75-1.0: pitch down (leveling)
        """
        
        # Sustained forward velocity
        vx = self.vx_forward
        
        # Pitch rate: derivative of pitch angle with respect to time
        # pitch(t) = A * sin(2π * f_pitch * t)
        # pitch_rate(t) = A * 2π * f_pitch * cos(2π * f_pitch * t)
        pitch_angular_freq = 2 * np.pi * self.freq * self.pitch_freq_multiplier
        pitch_rate = self.pitch_amplitude * pitch_angular_freq * np.cos(pitch_angular_freq * self.t)
        
        # Set velocity commands
        self.vel_world = np.array([vx, 0.0, 0.0])
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
        Compute foot position in BODY frame following a vertical circular trajectory.
        
        Each leg rotates continuously in a vertical circle perpendicular to the body axis.
        Circle parametrization:
        - angle = 2π * leg_phase
        - x_offset = radius * sin(angle)  (forward-backward motion)
        - z_offset = -radius * cos(angle) (up-down motion)
        
        At leg_phase = 0.0: angle = 0, foot at top front (x = 0, z = +radius)
        At leg_phase = 0.25: angle = π/2, foot at forward mid (x = +radius, z = 0)
        At leg_phase = 0.5: angle = π, foot at bottom rear (x = 0, z = -radius)
        At leg_phase = 0.75: angle = 3π/2, foot at rearward mid (x = -radius, z = 0)
        """
        
        # Apply phase offset for leg group coordination
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Get base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Circle center in body frame (relative to base foot position)
        circle_center_x = foot[0] + self.circle_center_offset_x
        circle_center_z = foot[2] + self.circle_center_offset_z
        
        # Parametric angle for circular motion
        angle = 2 * np.pi * leg_phase
        
        # Circular trajectory offsets
        # For propeller motion: top of circle is forward, bottom is rearward
        x_offset = self.circle_radius * np.sin(angle)
        z_offset = -self.circle_radius * np.cos(angle)
        
        # Apply circular trajectory to foot position
        foot[0] = circle_center_x + x_offset
        foot[2] = circle_center_z + z_offset
        
        # Y-coordinate (lateral) remains unchanged
        
        return foot