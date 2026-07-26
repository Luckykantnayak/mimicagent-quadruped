from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CLOVERLEAF_TRACE_MotionGenerator(BaseMotionGenerator):
    """
    Cloverleaf trace skill: robot traces a four-lobed cloverleaf pattern on the ground
    by coordinating base velocity changes with yaw rotation and synchronized leg reaching.
    All four feet maintain ground contact throughout the entire motion.
    
    Phase structure:
      [0.0, 0.25]: Lobe 1 - rightward curve, clockwise yaw, stance widens right
      [0.25, 0.5]: Lobe 2 - leftward curve, counter-clockwise yaw, stance widens left
      [0.5, 0.75]: Lobe 3 - forward-right diagonal, clockwise yaw, diagonal stance
      [0.75, 1.0]: Lobe 4 - backward-left return, counter-clockwise yaw, return to neutral
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.4  # Slower frequency for smooth cloverleaf tracing
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity command storage
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Motion parameters for cloverleaf lobes
        self.vx_max = 0.6  # Maximum forward velocity
        self.vy_max = 0.8  # Maximum lateral velocity
        self.yaw_rate_max = 0.9  # Maximum yaw rate (rad/s)
        
        # Stance modulation parameters
        self.stance_extension_max = 0.15  # Maximum outward foot reach
        
    def update_base_motion(self, phase, dt):
        """
        Update base motion to trace cloverleaf pattern.
        Each sub-phase corresponds to one petal lobe with coordinated
        linear and angular velocities.
        """
        
        # Determine which lobe we're in and compute smooth velocity profiles
        if phase < 0.25:
            # Lobe 1: Rightward curve with clockwise yaw
            sub_phase = phase / 0.25
            smooth = self._smooth_interpolant(sub_phase)
            
            vx = 0.3 * self.vx_max * smooth
            vy = self.vy_max * smooth
            yaw_rate = self.yaw_rate_max * smooth
            
        elif phase < 0.5:
            # Lobe 2: Leftward curve with counter-clockwise yaw
            sub_phase = (phase - 0.25) / 0.25
            smooth = self._smooth_interpolant(sub_phase)
            
            vx = 0.3 * self.vx_max * (1.0 - smooth * 0.5)
            vy = -self.vy_max * smooth
            yaw_rate = -self.yaw_rate_max * smooth
            
        elif phase < 0.75:
            # Lobe 3: Forward-right diagonal with clockwise yaw
            sub_phase = (phase - 0.5) / 0.25
            smooth = self._smooth_interpolant(sub_phase)
            
            vx = self.vx_max * smooth
            vy = 0.6 * self.vy_max * smooth
            yaw_rate = 0.8 * self.yaw_rate_max * smooth
            
        else:
            # Lobe 4: Backward-left return with counter-clockwise yaw
            sub_phase = (phase - 0.75) / 0.25
            smooth = self._smooth_interpolant(sub_phase)
            
            vx = -0.7 * self.vx_max * smooth
            vy = -0.7 * self.vy_max * smooth
            yaw_rate = -self.yaw_rate_max * smooth
        
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
        Compute foot position in body frame with stance width modulation
        to emphasize petal shapes. All feet remain in contact throughout.
        
        Diagonal pairs (FL+RR, FR+RL) coordinate reaching patterns:
        - Rightward lobes: FL+RR extend outward right
        - Leftward lobes: FR+RL extend outward left
        - Diagonal lobe: FL+RR extend forward-right
        - Return lobe: FR+RL extend backward-left
        """
        
        # Start from base position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg group for coordination
        is_fl_rr_group = leg_name.startswith('FL') or leg_name.startswith('RR')
        is_fr_rl_group = leg_name.startswith('FR') or leg_name.startswith('RL')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Compute stance modulation based on phase and leg
        if phase < 0.25:
            # Lobe 1: Rightward - FL+RR group leads, extend right
            sub_phase = phase / 0.25
            smooth = self._smooth_interpolant(sub_phase)
            
            if is_fl_rr_group:
                # Strong right extension
                foot[1] += self.stance_extension_max * smooth * (1.0 if leg_name.startswith('FL') else 0.8)
                if is_front:
                    foot[0] += 0.05 * smooth  # Slight forward reach
                else:
                    foot[0] -= 0.03 * smooth  # Slight backward reach
            else:
                # Moderate right extension for FR+RL
                foot[1] += 0.5 * self.stance_extension_max * smooth
                
        elif phase < 0.5:
            # Lobe 2: Leftward - FR+RL group leads, extend left
            sub_phase = (phase - 0.25) / 0.25
            smooth = self._smooth_interpolant(sub_phase)
            
            # Transition: retract from right then extend left
            retract_phase = min(sub_phase * 2.0, 1.0)
            extend_phase = max((sub_phase - 0.5) * 2.0, 0.0)
            
            if is_fr_rl_group:
                # Strong left extension
                foot[1] -= self.stance_extension_max * extend_phase * (1.0 if leg_name.startswith('FR') else 0.8)
                if is_front:
                    foot[0] += 0.05 * extend_phase  # Slight forward reach
                else:
                    foot[0] -= 0.03 * extend_phase  # Slight backward reach
            else:
                # FL+RR retract then moderate left
                foot[1] += self.stance_extension_max * (1.0 - retract_phase) * 0.5
                foot[1] -= 0.4 * self.stance_extension_max * extend_phase
                
        elif phase < 0.75:
            # Lobe 3: Forward-right diagonal - FL+RR group leads
            sub_phase = (phase - 0.5) / 0.25
            smooth = self._smooth_interpolant(sub_phase)
            
            if is_fl_rr_group:
                # Strong diagonal forward-right extension
                if leg_name.startswith('FL'):
                    foot[0] += 0.12 * smooth  # Strong forward
                    foot[1] += 0.9 * self.stance_extension_max * smooth  # Strong right
                else:  # RR
                    foot[0] += 0.05 * smooth  # Moderate forward
                    foot[1] += 0.7 * self.stance_extension_max * smooth  # Right
            else:
                # FR+RL moderate support
                foot[0] += 0.03 * smooth
                foot[1] += 0.3 * self.stance_extension_max * smooth
                
        else:
            # Lobe 4: Backward-left return - FR+RL group leads
            sub_phase = (phase - 0.75) / 0.25
            smooth = self._smooth_interpolant(sub_phase)
            
            # Return to neutral: retract extensions
            return_smooth = 1.0 - smooth
            
            if is_fr_rl_group:
                # Strong backward-left extension then return
                if leg_name.startswith('FR'):
                    foot[0] -= 0.05 * return_smooth
                    foot[1] -= 0.8 * self.stance_extension_max * return_smooth
                else:  # RL
                    foot[0] -= 0.08 * return_smooth  # Backward
                    foot[1] -= 0.9 * self.stance_extension_max * return_smooth  # Left
            else:
                # FL+RR return to neutral
                foot[0] += 0.03 * return_smooth
                foot[1] += 0.3 * self.stance_extension_max * return_smooth
        
        return foot
    
    def _smooth_interpolant(self, t):
        """
        Smooth interpolation function using cubic easing.
        Maps [0,1] -> [0,1] with zero derivatives at endpoints.
        """
        return 3 * t**2 - 2 * t**3