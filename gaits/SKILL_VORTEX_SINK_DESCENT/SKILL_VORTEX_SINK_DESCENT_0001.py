from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_VORTEX_SINK_DESCENT_MotionGenerator(BaseMotionGenerator):
    """
    Vortex Sink Descent: Descending spiral motion with continuous four-foot ground contact.
    
    - Base executes constant yaw rotation with decreasing forward velocity to create tightening spiral
    - Continuous downward z-velocity lowers base height
    - All four legs synchronously retract inward in body frame as spiral tightens
    - No aerial phase - all feet maintain ground contact throughout
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.3  # Slow cycle for smooth controlled descent
        
        # Store initial foot positions (wide stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compute initial radial distances for each foot
        self.initial_radial_distances = {}
        for leg in self.leg_names:
            pos = self.base_feet_pos_body[leg]
            self.initial_radial_distances[leg] = np.sqrt(pos[0]**2 + pos[1]**2)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity parameters for spiral trajectory
        self.vx_initial = 0.8  # Initial forward velocity (wide radius)
        self.vx_final = 0.15   # Final forward velocity (tight radius)
        self.yaw_rate = 1.5    # Constant yaw rate for continuous rotation
        
        # Descent parameters
        self.vz_descent = -0.25  # Downward velocity during descent
        self.total_descent = 0.3  # Total height loss over full cycle
        
        # Leg retraction parameters
        self.retraction_start_phase = 0.0
        self.retraction_end_phase = 0.85
        self.min_retraction_factor = 0.35  # Legs retract to 35% of initial radius at core

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands to create descending spiral trajectory.
        
        Phase structure:
        - [0.0, 0.25]: Wide spiral initiation with nominal height
        - [0.25, 0.5]: Active descent and radius contraction
        - [0.5, 0.75]: Deep spiral compression
        - [0.75, 1.0]: Minimum vortex core with tapering descent
        """
        
        # Forward velocity decreases smoothly to tighten spiral radius
        # Using cosine interpolation for smooth transition
        alpha = np.cos(np.pi * phase) * 0.5 + 0.5  # 1 -> 0 as phase: 0 -> 1
        vx = self.vx_initial * alpha + self.vx_final * (1 - alpha)
        
        # Vertical velocity for descent
        if phase < 0.25:
            # Initial phase: minimal descent
            vz = -0.05
        elif phase < 0.75:
            # Active descent phase
            vz = self.vz_descent
        else:
            # Taper descent toward end of cycle
            taper = (1.0 - phase) / 0.25
            vz = self.vz_descent * taper
        
        # Constant yaw rate throughout for continuous rotation
        yaw_rate = self.yaw_rate
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, 0.0, vz])
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
        Compute foot position in body frame with synchronized inward retraction.
        
        All legs maintain ground contact while moving radially inward toward
        body centerline as spiral tightens.
        """
        
        # Get base foot position
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Compute retraction factor based on phase
        if phase < self.retraction_start_phase:
            retraction_factor = 1.0  # No retraction
        elif phase > self.retraction_end_phase:
            retraction_factor = self.min_retraction_factor  # Full retraction
        else:
            # Smooth retraction using cosine interpolation
            progress = (phase - self.retraction_start_phase) / (self.retraction_end_phase - self.retraction_start_phase)
            alpha = np.cos(np.pi * progress) * 0.5 + 0.5  # 1 -> 0
            retraction_factor = alpha * 1.0 + (1 - alpha) * self.min_retraction_factor
        
        # Apply radial retraction in x-y plane
        foot_pos = base_pos.copy()
        foot_pos[0] *= retraction_factor
        foot_pos[1] *= retraction_factor
        
        # Z-position adjustment: legs extend downward to maintain ground contact as base descends
        # Compensate for base height loss to keep foot at ground level
        if phase < 0.25:
            z_compensation = 0.0
        else:
            # Approximate base descent based on velocity integration
            descent_phase = min(phase, 0.85)
            z_compensation = 0.1 * descent_phase  # Extend legs downward
        
        foot_pos[2] -= z_compensation
        
        return foot_pos