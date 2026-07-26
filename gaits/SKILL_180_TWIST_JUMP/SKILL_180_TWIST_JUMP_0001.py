from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_180_TWIST_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    180-degree twist jump with synchronized leg motion.
    
    Phase structure:
      0.00-0.15: Crouch and preload (all feet grounded, body lowers)
      0.15-0.25: Explosive takeoff (rapid vertical velocity, yaw rate initiates)
      0.25-0.65: Airborne rotation with tucked legs (all feet airborne, max yaw rate)
      0.65-0.75: Pre-landing extension (legs extend, yaw rate decelerates)
      0.75-1.00: Landing and absorption (all feet grounded, velocities zero out)
    
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
        
        # Motion parameters
        # Crouch depth during preload phase
        self.crouch_depth = 0.12
        
        # Leg tuck parameters: how much to retract inward and upward during airborne phase
        self.tuck_inward_factor = 0.5  # reduce lateral distance by 50%
        self.tuck_upward_height = 0.15  # lift feet upward in body frame
        
        # Vertical velocity during takeoff
        self.takeoff_vz = 2.0  # m/s upward
        
        # Yaw rate during rotation (tuned so integrated yaw = π over airborne duration)
        # Airborne phase duration ≈ 0.5 phase units (0.15 to 0.75)
        # To achieve π radians in 0.5 seconds (at freq=1.0): yaw_rate_peak = π / 0.4 ≈ 7.85 rad/s
        self.yaw_rate_peak = 8.0  # rad/s
        
        # Gravity for ballistic trajectory
        self.gravity = 9.81  # m/s^2
        
        # Landing compression depth
        self.landing_compression = 0.10

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Kinematic prescription of vertical and yaw motion.
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
            vz = -0.8 * np.sin(np.pi * progress)  # negative = downward
            # Small yaw preload
            yaw_rate = 0.5 * progress
        
        # Phase 0.15-0.25: Explosive takeoff
        elif phase < 0.25:
            progress = (phase - 0.15) / 0.1
            # Rapid upward velocity with smooth ramp
            vz = self.takeoff_vz * np.sin(np.pi * 0.5 * progress)
            # Rapidly increasing yaw rate
            yaw_rate = self.yaw_rate_peak * progress
        
        # Phase 0.25-0.65: Airborne rotation (ballistic + sustained yaw)
        elif phase < 0.65:
            progress = (phase - 0.25) / 0.4
            # Ballistic trajectory: initial upward velocity decays under gravity
            # Approximate parabolic motion
            time_in_air = progress * 0.4  # time in seconds at freq=1.0
            vz = self.takeoff_vz - self.gravity * time_in_air
            # Sustained peak yaw rate
            yaw_rate = self.yaw_rate_peak
        
        # Phase 0.65-0.75: Pre-landing (descending, yaw deceleration)
        elif phase < 0.75:
            progress = (phase - 0.65) / 0.1
            # Downward velocity continues (late ballistic phase)
            vz = -1.5
            # Yaw rate ramps down to zero
            yaw_rate = self.yaw_rate_peak * (1.0 - progress)
        
        # Phase 0.75-1.0: Landing and absorption
        else:
            progress = (phase - 0.75) / 0.25
            # Smooth deceleration to zero vertical velocity
            vz = -1.5 * (1.0 - progress) * np.cos(np.pi * 0.5 * progress)
            # Yaw rate brought to zero
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
        
        # Determine if front or rear leg (affects fore-aft shift during crouch)
        is_front = leg_name.startswith('F')
        
        # Phase 0.0-0.15: Crouch
        if phase < 0.15:
            progress = phase / 0.15
            # Smooth crouch: body lowers, so foot appears to move down in body frame
            foot[2] -= self.crouch_depth * np.sin(np.pi * 0.5 * progress)
            # Slight fore-aft shift to maintain balance
            if is_front:
                foot[0] += 0.02 * progress
            else:
                foot[0] -= 0.02 * progress
        
        # Phase 0.15-0.25: Takeoff (feet begin to lift and retract)
        elif phase < 0.25:
            progress = (phase - 0.15) / 0.1
            # Feet start retracting inward and upward as body accelerates up
            lateral_retract = self.tuck_inward_factor * progress
            foot[1] *= (1.0 - lateral_retract)  # move toward centerline
            foot[2] += self.tuck_upward_height * progress - self.crouch_depth
        
        # Phase 0.25-0.65: Airborne tuck
        elif phase < 0.65:
            progress = (phase - 0.25) / 0.4
            # Full tuck: feet pulled inward and upward
            foot[1] *= (1.0 - self.tuck_inward_factor)
            foot[2] += self.tuck_upward_height
            # Smooth tuck profile using sinusoid
            tuck_modulation = np.sin(np.pi * progress)
            foot[2] += 0.03 * tuck_modulation  # slight additional upward during mid-flight
        
        # Phase 0.65-0.75: Pre-landing extension
        elif phase < 0.75:
            progress = (phase - 0.65) / 0.1
            # Legs extend back outward and downward toward landing positions
            lateral_extend = self.tuck_inward_factor * (1.0 - progress)
            foot[1] = base_foot[1] * (1.0 - lateral_extend)
            # Height transitions from tucked to nominal
            height_from_tuck = self.tuck_upward_height * (1.0 - progress)
            foot[2] = base_foot[2] + height_from_tuck
        
        # Phase 0.75-1.0: Landing and absorption
        else:
            progress = (phase - 0.75) / 0.25
            # Feet return to nominal lateral position
            foot[1] = base_foot[1]
            foot[0] = base_foot[0]
            # Body lowers as legs compress to absorb impact
            compression = self.landing_compression * np.sin(np.pi * 0.5 * progress)
            foot[2] = base_foot[2] - compression
        
        return foot