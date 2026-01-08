from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SERPENTINE_BACKWARD_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Serpentine backward drift motion generator.
    
    The robot moves backward in a serpentine S-pattern by:
    - Maintaining constant backward velocity (negative vx)
    - Oscillating yaw rate to create body rotation (right then left)
    - Oscillating lateral velocity (left then right) synchronized with yaw
    - Alternating lateral leg extensions to kinematically produce drift effect
    
    All four feet remain in contact throughout the motion (stance-phase drift).
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Cycle frequency for smooth serpentine motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.backward_speed = -0.5  # Constant backward velocity (negative vx)
        self.lateral_drift_amp = 0.3  # Amplitude of lateral velocity oscillation
        self.yaw_rate_amp = 0.8  # Amplitude of yaw rate oscillation
        
        # Leg extension parameters (in body frame)
        self.lateral_extension_amp = 0.08  # How far legs extend laterally
        self.backward_offset_amp = 0.04  # Slight backward offset during extension
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (world frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with constant backward velocity and oscillating lateral/yaw.
        
        Phase breakdown:
        [0.0-0.25]: curve right (positive yaw), drift left (negative vy)
        [0.25-0.5]: transition (yaw and vy cross zero)
        [0.5-0.75]: curve left (negative yaw), drift right (positive vy)
        [0.75-1.0]: transition (yaw and vy return to starting values)
        """
        
        # Constant backward velocity
        vx = self.backward_speed
        
        # Lateral velocity oscillation (sinusoidal, phase-shifted for serpentine)
        # At phase=0: vy should be negative (left drift)
        # At phase=0.5: vy should be positive (right drift)
        vy = -self.lateral_drift_amp * np.cos(2 * np.pi * phase)
        
        # No vertical velocity
        vz = 0.0
        
        # Yaw rate oscillation (synchronized with lateral drift)
        # At phase=0: yaw_rate positive (rotating right)
        # At phase=0.5: yaw_rate negative (rotating left)
        yaw_rate = self.yaw_rate_amp * np.sin(2 * np.pi * phase)
        
        # No roll or pitch rate
        roll_rate = 0.0
        pitch_rate = 0.0
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
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
        Compute foot position in body frame with alternating lateral extensions.
        
        Left legs (FL, RL): extend left during [0.0-0.25], neutral during [0.5-0.75]
        Right legs (FR, RR): neutral during [0.0-0.25], extend right during [0.5-0.75]
        Transitions occur during [0.25-0.5] and [0.75-1.0]
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left or right leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        if is_left_leg:
            # Left legs: extend during [0.0-0.25], retract during [0.25-0.5], neutral during [0.5-1.0]
            if phase < 0.25:
                # Extension phase: smooth ramp from 0 to max extension
                progress = phase / 0.25
                extension_factor = self._smooth_step(progress)
                
                # Extend left (negative y in body frame)
                foot[1] -= self.lateral_extension_amp * extension_factor
                # Slight backward offset
                foot[0] -= self.backward_offset_amp * extension_factor
                
            elif phase < 0.5:
                # Retraction transition phase
                progress = (phase - 0.25) / 0.25
                retraction_factor = 1.0 - self._smooth_step(progress)
                
                foot[1] -= self.lateral_extension_amp * retraction_factor
                foot[0] -= self.backward_offset_amp * retraction_factor
                
            elif phase < 0.75:
                # Neutral stance during right legs' extension
                pass
                
            else:
                # Pre-positioning for next cycle
                progress = (phase - 0.75) / 0.25
                extension_factor = self._smooth_step(progress)
                
                foot[1] -= self.lateral_extension_amp * extension_factor
                foot[0] -= self.backward_offset_amp * extension_factor
        
        elif is_right_leg:
            # Right legs: neutral during [0.0-0.25], pre-position during [0.25-0.5], extend during [0.5-0.75], retract during [0.75-1.0]
            if phase < 0.25:
                # Neutral stance during left legs' extension
                pass
                
            elif phase < 0.5:
                # Pre-positioning for extension
                progress = (phase - 0.25) / 0.25
                extension_factor = self._smooth_step(progress)
                
                # Extend right (positive y in body frame)
                foot[1] += self.lateral_extension_amp * extension_factor
                # Slight backward offset
                foot[0] -= self.backward_offset_amp * extension_factor
                
            elif phase < 0.75:
                # Full extension phase
                progress = (phase - 0.5) / 0.25
                # Maintain extension with potential slight increase
                extension_factor = 1.0
                
                foot[1] += self.lateral_extension_amp * extension_factor
                foot[0] -= self.backward_offset_amp * extension_factor
                
            else:
                # Retraction transition phase
                progress = (phase - 0.75) / 0.25
                retraction_factor = 1.0 - self._smooth_step(progress)
                
                foot[1] += self.lateral_extension_amp * retraction_factor
                foot[0] -= self.backward_offset_amp * retraction_factor
        
        return foot
    
    def _smooth_step(self, x):
        """
        Smoothstep function for smooth transitions.
        Maps [0,1] -> [0,1] with zero derivatives at endpoints.
        """
        return x * x * (3.0 - 2.0 * x)