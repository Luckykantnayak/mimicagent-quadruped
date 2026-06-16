from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_STARFISH_PULSE_TURN_MotionGenerator(BaseMotionGenerator):
    """
    Starfish Pulse Turn: In-place yaw rotation with synchronized radial leg pulsing.
    
    All four legs simultaneously extend and retract radially while making small
    coordinated sweeping motions to generate yaw rotation. Creates a starfish-like
    visual effect with rhythmic pulsing synchronized to the turning motion.
    
    Phase breakdown:
      [0.0, 0.3]: Radial expansion - legs extend outward, yaw ramps up
      [0.3, 0.5]: Extended hold - max radius maintained, peak yaw velocity
      [0.5, 0.8]: Radial contraction - legs retract inward, yaw decelerates
      [0.8, 1.0]: Consolidation - min radius, prepare for next cycle
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8
        
        # Base foot positions (neutral stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compute radial vectors and base angles for each leg in body frame
        self.leg_radial_info = {}
        for leg in self.leg_names:
            base_pos = self.base_feet_pos_body[leg]
            radial_dist = np.sqrt(base_pos[0]**2 + base_pos[1]**2)
            radial_angle = np.arctan2(base_pos[1], base_pos[0])
            self.leg_radial_info[leg] = {
                'base_radial_dist': radial_dist,
                'base_angle': radial_angle,
                'base_z': base_pos[2]
            }
        
        # Motion parameters (reduced for kinematic feasibility)
        self.radial_extension_amplitude = 0.06  # Reduced from 0.12m
        self.angular_sweep_amplitude = 0.25  # Small oscillatory sweep in radians (~14 degrees)
        self.max_yaw_rate = 0.7  # Reduced from 1.2 rad/s
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        Yaw rate profile:
          - [0.0, 0.3]: Ramp up from low to peak
          - [0.3, 0.5]: Hold at peak
          - [0.5, 0.8]: Ramp down from peak to moderate
          - [0.8, 1.0]: Further decelerate to low
        
        No vertical velocity - base height remains constant.
        """
        
        # Compute yaw rate based on phase with smooth transitions
        if phase < 0.3:
            # Expansion: ramp up yaw rate
            progress = phase / 0.3
            smooth_progress = self._smooth_step(progress)
            yaw_rate = 0.3 * self.max_yaw_rate + (self.max_yaw_rate - 0.3 * self.max_yaw_rate) * smooth_progress
        elif phase < 0.5:
            # Extended hold: peak yaw rate
            yaw_rate = self.max_yaw_rate
        elif phase < 0.8:
            # Contraction: ramp down yaw rate
            progress = (phase - 0.5) / 0.3
            smooth_progress = self._smooth_step(progress)
            yaw_rate = self.max_yaw_rate - (self.max_yaw_rate - 0.4 * self.max_yaw_rate) * smooth_progress
        else:
            # Consolidation: further decelerate
            progress = (phase - 0.8) / 0.2
            smooth_progress = self._smooth_step(progress)
            yaw_rate = 0.4 * self.max_yaw_rate - (0.4 * self.max_yaw_rate - 0.3 * self.max_yaw_rate) * smooth_progress
        
        # No vertical motion - keep base at constant height
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
        Compute foot position in body frame combining:
        1. Radial expansion/contraction (primary pulsing motion)
        2. Small oscillatory angular sweep for yaw generation
        
        All legs move in perfect synchronization.
        Feet remain at constant body-frame z to maintain ground contact.
        """
        
        leg_info = self.leg_radial_info[leg_name]
        base_radial_dist = leg_info['base_radial_dist']
        base_angle = leg_info['base_angle']
        base_z = leg_info['base_z']
        
        # Compute radial extension factor based on phase with smooth transitions
        if phase < 0.3:
            # Expansion: smoothly increase from 0 to 1
            progress = phase / 0.3
            radial_factor = self._smooth_step(progress)
        elif phase < 0.5:
            # Extended hold: maintain max extension
            radial_factor = 1.0
        elif phase < 0.8:
            # Contraction: smoothly decrease from 1 to 0
            progress = (phase - 0.5) / 0.3
            radial_factor = 1.0 - self._smooth_step(progress)
        else:
            # Consolidation: maintain min extension
            radial_factor = 0.0
        
        # Current radial distance from body center
        radial_dist = base_radial_dist + self.radial_extension_amplitude * radial_factor
        
        # Small oscillatory angular sweep for yaw generation
        # Oscillates around base_angle rather than accumulating rotation
        # Clockwise sweep: negative angular offset when sweeping forward in phase
        angular_offset = -self.angular_sweep_amplitude * np.sin(2.0 * np.pi * phase)
        
        # Total angle remains constrained near base_angle
        total_angle = base_angle + angular_offset
        
        # Compute x, y position from radial distance and angle
        x = radial_dist * np.cos(total_angle)
        y = radial_dist * np.sin(total_angle)
        
        # Z position remains constant at base_z for ground contact
        z = base_z
        
        return np.array([x, y, z])
    
    def _smooth_step(self, t):
        """
        Smooth interpolation function (smoothstep) for continuous velocity.
        Maps [0,1] -> [0,1] with zero derivatives at endpoints.
        """
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)