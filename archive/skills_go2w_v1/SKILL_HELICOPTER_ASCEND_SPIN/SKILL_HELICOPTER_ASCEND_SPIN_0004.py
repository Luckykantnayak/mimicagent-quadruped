from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_HELICOPTER_ASCEND_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Helicopter ascending spin skill.
    
    - All four legs extend horizontally outward like rotor blades
    - Base rises vertically while spinning rapidly in yaw
    - Fully aerial maneuver with no ground contacts
    - Synchronized leg motion for balanced rotation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (tucked/neutral configuration)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Vertical motion parameters - conservative to respect height envelope
        self.initial_height_offset = 0.10   # Reduced from 0.18 to provide headroom
        self.vz_ascent_max = 0.28           # Reduced from 0.5 to limit integrated displacement
        
        # Yaw spin parameters
        self.yaw_rate_max = 8.0             # Peak yaw angular velocity (rad/s)
        self.yaw_rate_min = 1.0             # Minimum yaw rate during spin-down
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.initial_height_offset])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Rotor blade extension strategy: incremental offsets from base positions
        # Rather than absolute radial distances, apply modest delta to respect workspace
        self.rotor_positions = {}
        for leg in leg_names:
            base_pos = self.base_feet_pos_body[leg].copy()
            
            # Compute radial direction from body center in XY plane
            radial_xy = np.array([base_pos[0], base_pos[1]])
            radial_norm = np.linalg.norm(radial_xy)
            if radial_norm > 1e-6:
                radial_unit = radial_xy / radial_norm
            else:
                # Fallback radial directions if base position at origin
                if leg.startswith('FL'):
                    radial_unit = np.array([1.0, 1.0]) / np.sqrt(2)
                elif leg.startswith('FR'):
                    radial_unit = np.array([1.0, -1.0]) / np.sqrt(2)
                elif leg.startswith('RL'):
                    radial_unit = np.array([-1.0, 1.0]) / np.sqrt(2)
                else:  # RR
                    radial_unit = np.array([-1.0, -1.0]) / np.sqrt(2)
            
            # Incremental extension approach: apply modest outward shift from base position
            # Front legs slightly more conservative
            if leg.startswith('FL') or leg.startswith('FR'):
                delta_radial = 0.06  # Moderate outward extension
            else:
                delta_radial = 0.07  # Slightly more for rear legs
            
            # Vertical offset reduced to stay within joint limits
            delta_vertical = -0.025
            
            # Compute rotor position as base position plus incremental offset
            rotor_x = base_pos[0] + radial_unit[0] * delta_radial
            rotor_y = base_pos[1] + radial_unit[1] * delta_radial
            rotor_z = base_pos[2] + delta_vertical
            
            self.rotor_positions[leg] = np.array([rotor_x, rotor_y, rotor_z])

    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent vertical and yaw velocities.
        
        Phase ranges (symmetric acceleration/deceleration for height control):
        - [0.0, 0.3]: Spin up and smooth ascent acceleration
        - [0.3, 0.5]: Peak spin and peak ascent velocity
        - [0.5, 0.8]: Spin sustained, smooth ascent deceleration to zero
        - [0.8, 1.0]: Spin down to minimum rate, maintain altitude
        """
        
        # Compute vertical velocity with symmetric rise profile
        if phase < 0.3:
            # Ramp up ascent velocity with smooth acceleration
            progress = phase / 0.3
            smooth_progress = self._smooth_step(progress)
            vz = self.vz_ascent_max * smooth_progress
        elif phase < 0.5:
            # Sustained ascent at peak velocity
            vz = self.vz_ascent_max
        elif phase < 0.8:
            # Decelerate vertical velocity smoothly to zero (symmetric with accel phase)
            progress = (phase - 0.5) / 0.3
            smooth_progress = self._smooth_step(progress)
            vz = self.vz_ascent_max * (1.0 - smooth_progress)
        else:
            # Final phase: zero vertical velocity to stabilize altitude
            vz = 0.0
        
        # Compute yaw rate based on phase
        if phase < 0.3:
            # Spin up: ramp from zero to max with smooth acceleration
            progress = phase / 0.3
            smooth_progress = self._smooth_step(progress)
            yaw_rate = self.yaw_rate_max * smooth_progress
        elif phase < 0.8:
            # Peak spin rate sustained through ascent
            yaw_rate = self.yaw_rate_max
        else:
            # Spin down: decelerate smoothly toward minimum
            progress = (phase - 0.8) / 0.2
            smooth_progress = self._smooth_step(progress)
            yaw_rate = self.yaw_rate_max * (1.0 - smooth_progress) + self.yaw_rate_min * smooth_progress
        
        # Set velocities in world frame
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
        Compute foot position in body frame for helicopter rotor motion.
        
        All legs move synchronously:
        - [0.0, 0.3]: Extend from tucked position to horizontal rotor configuration
        - [0.3, 0.9]: Hold extended rotor position
        - [0.9, 1.0]: Retract back toward tucked position
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        rotor_pos = self.rotor_positions[leg_name].copy()
        
        if phase < 0.3:
            # Extension phase: smooth blend from base to rotor position
            progress = phase / 0.3
            # Use smoothstep for C1 continuity at phase boundaries
            blend = self._smooth_step(progress)
            foot = base_pos * (1.0 - blend) + rotor_pos * blend
            
        elif phase < 0.9:
            # Hold rotor configuration
            foot = rotor_pos.copy()
            
        else:
            # Retraction phase: blend from rotor back to base position
            progress = (phase - 0.9) / 0.1
            # Use smoothstep for C1 continuity
            blend = self._smooth_step(progress)
            foot = rotor_pos * (1.0 - blend) + base_pos * blend
        
        return foot
    
    def _smooth_step(self, x):
        """
        Smooth interpolation function (ease-in-ease-out).
        Maps [0,1] -> [0,1] with zero derivatives at endpoints.
        """
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)