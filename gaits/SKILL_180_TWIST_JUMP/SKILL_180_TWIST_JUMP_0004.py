from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_180_TWIST_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    180-degree twist jump with synchronized leg motion.
    
    Phase structure:
      0.00-0.15: Crouch and preload (all feet grounded, body lowers)
      0.15-0.25: Explosive takeoff (rapid vertical velocity, yaw rate initiates)
      0.25-0.55: Airborne rotation with tucked legs (all feet airborne, max yaw rate)
      0.55-0.70: Pre-landing extension (legs extend, yaw rate decelerates)
      0.70-1.00: Landing and absorption (all feet grounded, velocities zero out)
    
    Target: 180-degree yaw rotation integrated over the motion cycle.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.2
        
        # Store nominal foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.crouch_depth = 0.02
        
        self.tuck_inward_factor = 0.15
        self.tuck_upward_height = 0.04
        
        self.takeoff_vz = 3.  # m/s
        
        self.yaw_rate_peak = 10.5  # rad/s
        
        self.gravity = 9.81  # m/s^2
        
        self.peak_altitude = 0.55
        
        # Nominal base height - actively used as landing target
        self.nominal_base_height = 0.5
        
        # Track initial height to compute landing target
        self.initial_base_height = 0.0
        self.landing_target_height = 0.0
        self.height_initialized = False

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Kinematic prescription of vertical and yaw motion with controlled altitude.
        Landing phase ensures base returns to nominal height with feet on ground.
        """
        
        # Initialize landing target on first call
        if not self.height_initialized:
            self.initial_base_height = self.root_pos[2]
            self.landing_target_height = self.initial_base_height
            self.height_initialized = True
        
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.15: Crouch (downward velocity)
        if phase < 0.15:
            progress = phase / 0.15
            vz = -0.5 * np.sin(np.pi * progress)
            yaw_rate = 0.5 * progress
        
        # Phase 0.15-0.25: Explosive takeoff
        elif phase < 0.25:
            progress = (phase - 0.15) / 0.1
            vz = self.takeoff_vz * np.sin(np.pi * 0.5 * progress)
            yaw_rate = self.yaw_rate_peak * (progress ** 0.8)
        
        # Phase 0.25-0.55: Airborne rotation (balanced parabolic trajectory)
        elif phase < 0.55:
            progress = (phase - 0.25) / 0.30
            vz = self.takeoff_vz * (1.0 - 2.0 * progress)
            yaw_rate = self.yaw_rate_peak
        
        # Phase 0.55-0.70: Pre-landing (descending, yaw deceleration)
        elif phase < 0.70:
            progress = (phase - 0.55) / 0.15
            vz = -0.9 * (1.0 + 0.2 * progress)
            yaw_rate = self.yaw_rate_peak * (1.0 - progress) ** 2
        
        # Phase 0.70-1.0: Landing and absorption (position-aware velocity control)
        else:
            progress = (phase - 0.70) / 0.30
            
            # Compute height error relative to landing target
            height_error = self.root_pos[2] - self.landing_target_height
            
            # Position-aware velocity profile: decelerates as base approaches target height
            # If above target, continue descent; if at or below target, stabilize
            if height_error > 0.02:
                # Still above target: controlled descent with deceleration
                vz = -0.8 * (1.0 - progress) ** 2.5 * min(height_error / 0.1, 1.0)
            elif height_error > -0.01:
                # Near target: very gentle settling
                vz = -0.1 * (1.0 - progress)
            else:
                # Below target: slight upward correction to prevent penetration
                vz = 0.2 * min(abs(height_error), 0.05)
            
            yaw_rate = 0.0
        
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
        Compute foot position in body frame for given leg and phase.
        All four legs move synchronously through the motion phases.
        Feet extend to maintain ground contact during landing absorption.
        """
        
        # Get base foot position
        base_foot = self.base_feet_pos_body[leg_name].copy()
        foot = base_foot.copy()
        
        # Determine if front or rear leg
        is_front = leg_name.startswith('F')
        
        # Phase 0.0-0.15: Crouch
        if phase < 0.15:
            progress = phase / 0.15
            # Smooth crouch primarily through base motion, minimal foot adjustment
            foot[2] -= 0.3 * self.crouch_depth * (progress ** 1.5)
            if is_front:
                foot[0] += 0.015 * progress
            else:
                foot[0] -= 0.015 * progress
        
        # Phase 0.15-0.25: Takeoff (feet begin to lift and retract)
        elif phase < 0.25:
            progress = (phase - 0.15) / 0.1
            lateral_retract = self.tuck_inward_factor * (progress ** 1.2)
            foot[1] *= (1.0 - lateral_retract)
            crouch_offset = 0.3 * self.crouch_depth * (1.0 - progress)
            tuck_lift = self.tuck_upward_height * (progress ** 1.5)
            foot[2] = base_foot[2] - crouch_offset + tuck_lift
        
        # Phase 0.25-0.55: Airborne tuck
        elif phase < 0.55:
            progress = (phase - 0.25) / 0.30
            foot[1] *= (1.0 - self.tuck_inward_factor)
            foot[2] = base_foot[2] + self.tuck_upward_height
            tuck_modulation = 0.01 * np.sin(np.pi * progress)
            foot[2] += tuck_modulation
        
        # Phase 0.55-0.70: Pre-landing extension
        elif phase < 0.70:
            progress = (phase - 0.55) / 0.15
            lateral_extend = self.tuck_inward_factor * (1.0 - progress ** 1.5)
            foot[1] = base_foot[1] * (1.0 - lateral_extend)
            height_from_tuck = self.tuck_upward_height * (1.0 - progress ** 1.8)
            foot[2] = base_foot[2] + height_from_tuck
        
        # Phase 0.70-1.0: Landing and absorption
        else:
            progress = (phase - 0.70) / 0.30
            
            # Feet return to nominal lateral and fore-aft positions
            foot[1] = base_foot[1]
            foot[0] = base_foot[0]
            
            # Feet extend downward during absorption to maintain ground contact
            # As base settles, feet reach down to "seek" ground
            extension_depth = 0.08 * np.sin(np.pi * 0.5 * min(progress / 0.7, 1.0))
            foot[2] = base_foot[2] - extension_depth
        
        return foot