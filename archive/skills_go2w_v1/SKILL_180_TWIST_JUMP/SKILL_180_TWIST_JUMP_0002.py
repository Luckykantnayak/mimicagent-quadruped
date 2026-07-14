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
        self.freq = 1.0
        
        # Store nominal foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters - tuned for compact jump
        self.crouch_depth = 0.12
        
        # Reduced tuck parameters to avoid joint limit violations
        self.tuck_inward_factor = 0.30  # reduced from 0.5 to stay within reachable workspace
        self.tuck_upward_height = 0.09  # reduced from 0.15 to avoid extreme knee flexion
        
        # Reduced takeoff velocity for compact jump (lower peak altitude)
        self.takeoff_vz = 1.2  # reduced from 2.0 m/s
        
        # Yaw rate tuned for 180-degree rotation over shorter airborne duration
        # Airborne phase now 0.30 phase units (0.25 to 0.55)
        # Target: π radians in ~0.30 seconds at freq=1.0
        self.yaw_rate_peak = 10.5  # rad/s
        
        self.gravity = 9.81  # m/s^2
        
        # Peak altitude for parabolic trajectory control
        self.peak_altitude = 0.40  # target peak height in meters (compact jump)
        
        # Landing phase parameters
        self.landing_base_descent = 0.08  # base descends during absorption

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Kinematic prescription of vertical and yaw motion with controlled altitude.
        """
        
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.15: Crouch (downward velocity)
        if phase < 0.15:
            progress = phase / 0.15
            # Smooth downward velocity during crouch
            vz = -0.8 * np.sin(np.pi * progress)
            # Small yaw preload
            yaw_rate = 0.5 * progress
        
        # Phase 0.15-0.25: Explosive takeoff
        elif phase < 0.25:
            progress = (phase - 0.15) / 0.1
            # Smooth upward acceleration
            vz = self.takeoff_vz * np.sin(np.pi * 0.5 * progress)
            # Rapidly increasing yaw rate
            yaw_rate = self.yaw_rate_peak * (progress ** 0.8)
        
        # Phase 0.25-0.55: Airborne rotation (parabolic trajectory)
        elif phase < 0.55:
            progress = (phase - 0.25) / 0.30
            # Parabolic vertical velocity profile (up then down)
            # Peak at progress = 0.5, symmetric ascent/descent
            vz = self.takeoff_vz * (1.0 - 2.0 * progress) - 0.3
            # Sustained peak yaw rate
            yaw_rate = self.yaw_rate_peak
        
        # Phase 0.55-0.70: Pre-landing (descending, yaw deceleration)
        elif phase < 0.70:
            progress = (phase - 0.55) / 0.15
            # Controlled descent velocity
            vz = -1.1 * (1.0 + 0.3 * progress)
            # Yaw rate ramps down to zero smoothly
            yaw_rate = self.yaw_rate_peak * (1.0 - progress) ** 2
        
        # Phase 0.70-1.0: Landing and absorption
        else:
            progress = (phase - 0.70) / 0.30
            # Smooth deceleration to zero vertical velocity
            vz = -1.4 * (1.0 - progress) ** 2
            # Yaw rate at zero
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
        """
        
        # Get base foot position
        base_foot = self.base_feet_pos_body[leg_name].copy()
        foot = base_foot.copy()
        
        # Determine if front or rear leg
        is_front = leg_name.startswith('F')
        
        # Phase 0.0-0.15: Crouch
        if phase < 0.15:
            progress = phase / 0.15
            # Body lowers, feet move down relative to body
            foot[2] -= self.crouch_depth * (progress ** 1.5)
            # Slight fore-aft shift for balance
            if is_front:
                foot[0] += 0.02 * progress
            else:
                foot[0] -= 0.02 * progress
        
        # Phase 0.15-0.25: Takeoff (feet begin to lift and retract)
        elif phase < 0.25:
            progress = (phase - 0.15) / 0.1
            # Feet start retracting inward and upward smoothly
            lateral_retract = self.tuck_inward_factor * (progress ** 0.7)
            foot[1] *= (1.0 - lateral_retract)
            # Smooth transition from crouch to tuck height
            foot[2] = base_foot[2] - self.crouch_depth * (1.0 - progress) + self.tuck_upward_height * (progress ** 0.8)
        
        # Phase 0.25-0.55: Airborne tuck
        elif phase < 0.55:
            progress = (phase - 0.25) / 0.30
            # Full tuck maintained with smooth profile
            foot[1] *= (1.0 - self.tuck_inward_factor)
            foot[2] = base_foot[2] + self.tuck_upward_height
            # Slight modulation for natural motion
            tuck_modulation = 0.02 * np.sin(np.pi * progress)
            foot[2] += tuck_modulation
        
        # Phase 0.55-0.70: Pre-landing extension
        elif phase < 0.70:
            progress = (phase - 0.55) / 0.15
            # Legs extend back outward and downward smoothly
            lateral_extend = self.tuck_inward_factor * (1.0 - progress ** 0.6)
            foot[1] = base_foot[1] * (1.0 - lateral_extend)
            # Height transitions from tucked back to nominal
            height_from_tuck = self.tuck_upward_height * (1.0 - progress ** 0.8)
            foot[2] = base_foot[2] + height_from_tuck
        
        # Phase 0.70-1.0: Landing and absorption
        else:
            progress = (phase - 0.70) / 0.30
            # Feet return to nominal lateral and fore-aft positions
            foot[1] = base_foot[1]
            foot[0] = base_foot[0]
            # Feet stay at nominal height while base descends (absorption in base motion)
            # Small upward adjustment in body frame compensates for base descent
            absorption_compensation = self.landing_base_descent * (progress ** 1.2)
            foot[2] = base_foot[2] + absorption_compensation
        
        return foot