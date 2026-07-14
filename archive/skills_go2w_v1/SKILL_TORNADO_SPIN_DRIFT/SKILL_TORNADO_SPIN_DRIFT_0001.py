from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_TORNADO_SPIN_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Tornado Spin Drift: Rapid in-place yaw spin with dynamic leg radial modulation.
    
    Motion characteristics:
    - Continuous yaw rotation with sinusoidal acceleration/deceleration
    - All four legs synchronously retract/extend radially
    - Legs extended at phase 0.0 and 1.0, most retracted at phase ~0.625
    - Slight outward drift proportional to yaw rate (centrifugal effect)
    - All feet maintain continuous ground contact (stance throughout)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for full spin cycle visualization
        
        # Store base foot positions (extended stance configuration)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compute radial directions for each leg in body frame
        # FL: forward-left, FR: forward-right, RL: rear-left, RR: rear-right
        self.leg_radial_directions = {}
        for leg_name, pos in self.base_feet_pos_body.items():
            # Normalize xy direction from body center
            direction = pos.copy()
            direction[2] = 0  # Ignore vertical component for radial direction
            norm = np.linalg.norm(direction[:2])
            if norm > 1e-6:
                direction[:2] /= norm
            self.leg_radial_directions[leg_name] = direction
        
        # Radial extension parameters
        self.max_radial_extension = 1.0  # Multiplier for base position (fully extended)
        self.min_radial_extension = 0.4  # Multiplier for base position (tucked)
        
        # Yaw spin parameters
        self.max_yaw_rate = 4.0  # rad/s at peak spin
        
        # Drift parameters (centrifugal-like effect)
        self.drift_gain = 0.08  # Linear velocity per unit yaw rate
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def compute_radial_extension_factor(self, phase):
        """
        Compute radial extension factor as function of phase.
        
        Phase 0.0-0.25: Extended (1.0)
        Phase 0.25-0.5: Retracting (1.0 -> min)
        Phase 0.5-0.75: Tucked (min)
        Phase 0.75-1.0: Extending (min -> 1.0)
        
        Using smooth cosine interpolation for continuous motion.
        """
        if phase < 0.25:
            # Extended, constant
            return self.max_radial_extension
        elif phase < 0.5:
            # Retracting phase
            local_phase = (phase - 0.25) / 0.25
            # Smooth transition using cosine
            blend = 0.5 * (1 + np.cos(np.pi * local_phase))
            return self.min_radial_extension + (self.max_radial_extension - self.min_radial_extension) * blend
        elif phase < 0.75:
            # Tucked, constant at minimum
            return self.min_radial_extension
        else:
            # Extending phase
            local_phase = (phase - 0.75) / 0.25
            # Smooth transition using cosine
            blend = 0.5 * (1 + np.cos(np.pi * (1 - local_phase)))
            return self.min_radial_extension + (self.max_radial_extension - self.min_radial_extension) * blend

    def compute_yaw_rate(self, phase):
        """
        Compute yaw rate as function of phase.
        
        Smooth acceleration from 0 to peak, then smooth deceleration.
        Using sinusoidal profile for continuity.
        
        Phase 0.0-0.5: Accelerating (0 -> max)
        Phase 0.5-1.0: Decelerating (max -> 0)
        """
        if phase < 0.5:
            # Acceleration phase: sinusoidal ramp up
            local_phase = phase / 0.5
            return self.max_yaw_rate * np.sin(np.pi * local_phase / 2)
        else:
            # Deceleration phase: sinusoidal ramp down
            local_phase = (phase - 0.5) / 0.5
            return self.max_yaw_rate * np.cos(np.pi * local_phase / 2)

    def update_base_motion(self, phase, dt):
        """
        Update base pose with yaw spin and outward drift.
        
        Yaw rate follows smooth acceleration/deceleration profile.
        Linear drift in xy proportional to current yaw rate (centrifugal effect).
        """
        yaw_rate = self.compute_yaw_rate(phase)
        
        # Compute drift velocity (perpendicular to current heading, outward spiral)
        # Direction of drift rotates with current yaw angle
        current_yaw = quat_to_euler(self.root_quat)[2]
        drift_magnitude = self.drift_gain * abs(yaw_rate)
        
        # Drift direction: outward spiral (perpendicular to angular velocity)
        # Using time-varying direction to create spiral pattern
        drift_angle = current_yaw + np.pi / 2
        vx = drift_magnitude * np.cos(drift_angle)
        vy = drift_magnitude * np.sin(drift_angle)
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, 0.0])
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
        Compute foot position in body frame with synchronized radial modulation.
        
        All legs move radially in/out simultaneously along their respective diagonals.
        Maintains ground contact (z coordinate from base position).
        """
        # Get base foot position and radial direction
        base_pos = self.base_feet_pos_body[leg_name].copy()
        radial_dir = self.leg_radial_directions[leg_name].copy()
        
        # Compute current radial extension factor
        extension_factor = self.compute_radial_extension_factor(phase)
        
        # Apply radial scaling
        # Keep z at base height (constant ground contact)
        foot_pos = base_pos.copy()
        foot_pos[:2] = base_pos[:2] * extension_factor
        
        return foot_pos