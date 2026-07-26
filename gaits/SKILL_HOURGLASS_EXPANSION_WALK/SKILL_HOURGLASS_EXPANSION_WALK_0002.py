from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_HOURGLASS_EXPANSION_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Hourglass Expansion Walk: Forward walking gait with rhythmic stance width modulation.
    
    - All four legs converge inward (narrow stance) then expand outward (wide stance) synchronously
    - Base height rises during narrow stance, lowers during wide stance
    - Continuous forward velocity throughout cycle (peaks during narrow stance)
    - All feet remain in ground contact (stance-modulation gait, no swing phase)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Cycle frequency (Hz) - slower for smooth hourglass effect
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Stance width modulation parameters - constrained to kinematic workspace
        self.narrow_width_factor = 0.65  # Minimum lateral width (65% of nominal)
        self.wide_width_factor = 1.25    # Maximum lateral width (125% of nominal)
        
        # Forward motion parameters
        self.forward_velocity_min = 0.3   # Minimum forward velocity (m/s)
        self.forward_velocity_max = 0.8   # Maximum forward velocity during narrow stance (m/s)
        
        # Base height modulation parameters - reduced to maintain ground contact
        self.base_height_amplitude = 0.035  # Vertical oscillation amplitude (m)
        self.nominal_base_height = 0.0  # Will be set from initial position
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def compute_stance_width_factor(self, phase):
        """
        Compute stance width modulation factor as function of phase.
        Smooth continuous transition using single sinusoidal function.
        
        Returns factor in range [narrow_width_factor, wide_width_factor]
        """
        mid_width = (self.narrow_width_factor + self.wide_width_factor) / 2.0
        amplitude = (self.wide_width_factor - self.narrow_width_factor) / 2.0
        
        # Phase mapping: 0.0-0.4 converge (decreasing), 0.4-0.8 expand (increasing)
        # Offset by 0.4 so minimum occurs at phase=0.4, maximum at phase=0.8
        width_phase = phase - 0.2
        width_factor = mid_width - amplitude * np.cos(2.0 * np.pi * width_phase)
        
        return width_factor

    def compute_forward_velocity(self, phase):
        """
        Compute forward velocity (vx) as function of phase.
        Smooth continuous transition with peak during narrow stance.
        """
        mid_vel = (self.forward_velocity_min + self.forward_velocity_max) / 2.0
        amplitude = (self.forward_velocity_max - self.forward_velocity_min) / 2.0
        
        # Peak velocity at narrow stance (phase ~0.2-0.4), minimum at wide (phase ~0.6-0.8)
        # Offset by 0.5 so maximum occurs at phase=0.3, minimum at phase=0.8
        vel_phase = phase - 0.05
        velocity = mid_vel + amplitude * np.cos(2.0 * np.pi * vel_phase)
        
        return velocity

    def compute_base_height(self, phase):
        """
        Compute base height as direct position target.
        Height is inversely correlated with stance width:
        - Peak height during narrow stance (phase ~0.4)
        - Minimum height during wide stance (phase ~0.8)
        """
        # Use sinusoidal function: peak at phase=0.4, minimum at phase=0.8
        # Offset by 0.6 so cos() peak (phase=0) maps to phase=0.4
        height_phase = phase + 0.1
        height = self.nominal_base_height + self.base_height_amplitude * np.cos(2.0 * np.pi * height_phase)
        
        return height

    def compute_base_height_velocity(self, phase):
        """
        Compute vertical velocity as analytical derivative of height function.
        Ensures smooth continuous integration.
        """
        height_phase = phase + 0.1
        # Derivative of cos is -sin, scaled by angular frequency
        vz = -self.base_height_amplitude * 2.0 * np.pi * self.freq * np.sin(2.0 * np.pi * height_phase)
        
        return vz

    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent velocities.
        
        - Forward velocity modulated with phase (peak during narrow stance)
        - Vertical velocity coordinated inversely with stance width
        - No lateral or angular motion
        """
        vx = self.compute_forward_velocity(phase)
        vz = self.compute_base_height_velocity(phase)
        
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.zeros(3)  # No rotation
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_z_offset(self, width_factor):
        """
        Compute body-frame z-offset for foot position as function of stance width.
        During wide stance, feet must be commanded lower in body frame to maintain
        kinematic feasibility as lateral extension reduces vertical reach.
        """
        # Linear interpolation: narrow stance (0.65) -> z_offset ~ 0.0
        #                       wide stance (1.25) -> z_offset ~ -0.04
        normalized_width = (width_factor - self.narrow_width_factor) / (self.wide_width_factor - self.narrow_width_factor)
        normalized_width = np.clip(normalized_width, 0.0, 1.0)
        
        z_offset = -0.04 * normalized_width  # More negative as stance widens
        
        return z_offset

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with synchronized lateral modulation.
        
        All legs modulate their lateral positions (body-y) synchronously:
        - Converge inward during [0.0-0.4]
        - Expand outward during [0.4-0.8]
        - Begin convergence during [0.8-1.0]
        
        Vertical position (z) adjusted to compensate for lateral extension effects.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Get stance width modulation factor
        width_factor = self.compute_stance_width_factor(phase)
        
        # Modulate lateral position (body-y) symmetrically
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            # Left side: modulate positive body-y
            base_pos[1] = abs(self.base_feet_pos_body[leg_name][1]) * width_factor
        elif leg_name.startswith('FR') or leg_name.startswith('RR'):
            # Right side: modulate negative body-y
            base_pos[1] = -abs(self.base_feet_pos_body[leg_name][1]) * width_factor
        
        # Forward motion in body frame: smooth sinusoidal fore-aft motion
        forward_offset = 0.04 * np.sin(2.0 * np.pi * phase)
        base_pos[0] = self.base_feet_pos_body[leg_name][0] + forward_offset
        
        # Adjust z-coordinate in body frame to maintain kinematic feasibility
        z_offset = self.compute_foot_z_offset(width_factor)
        base_pos[2] = z_offset
        
        return base_pos

    def step(self, dt):
        """
        Update motion generator state by one timestep.
        """
        self.t += dt
        phase = (self.t * self.freq) % 1.0
        
        # Update base motion
        self.update_base_motion(phase, dt)
        
        # Compute foot positions in body frame
        foot_positions_body = {}
        for leg in self.leg_names:
            foot_positions_body[leg] = self.compute_foot_position_body_frame(leg, phase)
        
        return {
            'root_pos': self.root_pos.copy(),
            'root_quat': self.root_quat.copy(),
            'foot_positions_body': foot_positions_body,
            'vel_world': self.vel_world.copy(),
            'omega_world': self.omega_world.copy()
        }