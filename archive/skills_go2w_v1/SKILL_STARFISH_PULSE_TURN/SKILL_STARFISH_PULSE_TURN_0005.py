from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_STARFISH_PULSE_TURN_MotionGenerator(BaseMotionGenerator):
    """
    Starfish Pulse Turn: In-place yaw rotation with synchronized radial leg pulsing.
    
    All four legs simultaneously extend and retract radially while tracing clockwise
    circular arcs to generate continuous yaw rotation. Creates a starfish-like visual
    effect with rhythmic pulsing synchronized to the turning motion.
    
    Phase breakdown:
      [0.0, 0.3]: Radial expansion - legs extend outward, yaw ramps up
      [0.3, 0.5]: Extended hold - max radius maintained, peak yaw velocity
      [0.5, 0.8]: Radial contraction - legs retract inward, yaw decelerates
      [0.8, 1.0]: Consolidation - min radius, slight upward base motion
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Slower cycle for visible pulsing effect
        
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
        
        # Motion parameters
        self.radial_extension_amplitude = 0.12  # Radial expansion distance (m)
        self.circular_arc_radius = 0.08  # Radius of circular motion for yaw generation
        self.max_yaw_rate = 1.2  # Peak yaw rate (rad/s)
        self.base_lift_amplitude = 0.02  # Slight upward pulse during consolidation
        
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
          - [0.0, 0.3]: Ramp up from 0.4 to peak
          - [0.3, 0.5]: Hold at peak
          - [0.5, 0.8]: Ramp down from peak to moderate
          - [0.8, 1.0]: Further decelerate to low
        
        Vertical velocity:
          - [0.8, 1.0]: Slight upward motion for breathing effect
        """
        
        # Compute yaw rate based on phase
        if phase < 0.3:
            # Expansion: ramp up yaw rate
            progress = phase / 0.3
            yaw_rate = 0.4 * self.max_yaw_rate + (self.max_yaw_rate - 0.4 * self.max_yaw_rate) * progress
        elif phase < 0.5:
            # Extended hold: peak yaw rate
            yaw_rate = self.max_yaw_rate
        elif phase < 0.8:
            # Contraction: ramp down yaw rate
            progress = (phase - 0.5) / 0.3
            yaw_rate = self.max_yaw_rate - (self.max_yaw_rate - 0.5 * self.max_yaw_rate) * progress
        else:
            # Consolidation: further decelerate
            progress = (phase - 0.8) / 0.2
            yaw_rate = 0.5 * self.max_yaw_rate - (0.5 * self.max_yaw_rate - 0.3 * self.max_yaw_rate) * progress
        
        # Compute vertical velocity during consolidation phase
        if phase >= 0.8:
            progress = (phase - 0.8) / 0.2
            vz = self.base_lift_amplitude * np.sin(np.pi * progress) / dt if dt > 0 else 0.0
        else:
            vz = 0.0
        
        # Set velocity commands (no horizontal drift, only in-place rotation)
        self.vel_world = np.array([0.0, 0.0, vz])
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
        1. Radial expansion/contraction
        2. Clockwise circular motion for yaw generation
        
        All legs move in perfect synchronization (no phase offsets).
        """
        
        leg_info = self.leg_radial_info[leg_name]
        base_radial_dist = leg_info['base_radial_dist']
        base_angle = leg_info['base_angle']
        base_z = leg_info['base_z']
        
        # Compute radial extension factor based on phase
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
        
        # Current radial distance
        radial_dist = base_radial_dist + self.radial_extension_amplitude * radial_factor
        
        # Compute circular arc angle for yaw generation (clockwise = negative angle increment)
        # Accumulated angle over full phase cycle
        circular_angle = -2.0 * np.pi * phase
        
        # Combine base angle with circular motion angle
        total_angle = base_angle + circular_angle
        
        # Compute x, y position from radial distance and angle
        x = radial_dist * np.cos(total_angle)
        y = radial_dist * np.sin(total_angle)
        
        # Z position: base height with slight lift during consolidation
        if phase >= 0.8:
            progress = (phase - 0.8) / 0.2
            z_offset = self.base_lift_amplitude * np.sin(np.pi * progress)
        else:
            z_offset = 0.0
        
        z = base_z + z_offset
        
        return np.array([x, y, z])
    
    def _smooth_step(self, t):
        """
        Smooth interpolation function (smoothstep) for continuous velocity.
        Maps [0,1] -> [0,1] with zero derivatives at endpoints.
        """
        return t * t * (3.0 - 2.0 * t)