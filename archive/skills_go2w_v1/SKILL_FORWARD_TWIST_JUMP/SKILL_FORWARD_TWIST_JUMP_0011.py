from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_FORWARD_TWIST_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Forward jump with 180-degree yaw rotation during flight.
    
    Phase breakdown:
      [0.0, 0.2]: compression - all legs compress, body lowers
      [0.2, 0.4]: takeoff - explosive extension, body launches forward and upward
      [0.4, 0.7]: aerial_twist - body rotates 180 degrees in yaw, legs tuck
      [0.7, 0.9]: landing_preparation - rotation completes, legs extend
      [0.9, 1.0]: landing - all feet contact, legs compress to absorb impact
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Skill duration ~2 seconds
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Initialize base position to keep feet at ground level
        # Assume base is positioned such that initial foot z positions are at ground
        avg_foot_z = np.mean([v[2] for v in self.base_feet_pos_body.values()])
        self.initial_base_height = -avg_foot_z + 0.05  # Small clearance
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.initial_base_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Kinematic parameters - reduced for safety
        self.compression_depth = 0.10  # Reduced from 0.15
        self.tuck_factor = 0.4  # Reduced from 0.5
        
        # Velocity parameters - significantly reduced to stay within envelope
        self.takeoff_vx = 1.2  # Forward velocity (m/s)
        self.takeoff_vz = 1.1  # Reduced from 2.5 to stay under 0.68m ceiling
        
        # Yaw rotation parameters (180 degrees over 0.3 phase duration)
        self.yaw_rate_aerial = np.pi / (0.3 / self.freq)  # rad/s
        
        # Gravity approximation
        self.gravity = 9.81
        
    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on current phase.
        Velocities are tuned to maintain base height within safe envelope.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 1: Compression [0.0, 0.2]
        if phase < 0.2:
            # Controlled descent coordinated with leg compression
            # Max descent = compression_depth to keep feet grounded
            progress = phase / 0.2
            # Reduced magnitude: integrate over duration yields ~compression_depth
            vz = -0.35 * np.sin(progress * np.pi)
            
        # Phase 2: Takeoff [0.2, 0.4]
        elif phase < 0.4:
            # Explosive but controlled upward and forward acceleration
            progress = (phase - 0.2) / 0.2
            # Smooth acceleration profile
            accel_profile = np.sin(progress * np.pi)
            vx = self.takeoff_vx * accel_profile
            vz = self.takeoff_vz * accel_profile
            
        # Phase 3: Aerial twist [0.4, 0.7]
        elif phase < 0.7:
            # Ballistic trajectory with yaw rotation
            progress = (phase - 0.4) / 0.3
            
            # Forward velocity maintained
            vx = self.takeoff_vx * 0.85
            
            # Vertical velocity: parabolic arc (up then down)
            # Peak at mid-aerial phase
            apex_progress = 0.5
            if progress < apex_progress:
                # Ascending to apex
                vz = self.takeoff_vz * 0.4 * (1.0 - progress / apex_progress)
            else:
                # Descending from apex
                descent_progress = (progress - apex_progress) / (1.0 - apex_progress)
                vz = -self.gravity * 0.25 * descent_progress
            
            # Yaw rotation at constant rate
            yaw_rate = self.yaw_rate_aerial
            
        # Phase 4: Landing preparation [0.7, 0.9]
        elif phase < 0.9:
            # Descending toward ground, rotation complete
            progress = (phase - 0.7) / 0.2
            
            # Decelerate forward motion
            vx = self.takeoff_vx * 0.6 * (1.0 - progress)
            
            # Controlled descent with increasing rate
            vz = -self.gravity * 0.35 * (1.0 + progress * 0.5)
            
            yaw_rate = 0.0
            
        # Phase 5: Landing [0.9, 1.0]
        else:
            # Impact absorption - controlled descent with damping
            progress = (phase - 0.9) / 0.1
            
            # Gentle downward velocity to coordinate with leg compression
            # Allows base to settle as legs compress
            if progress < 0.5:
                vz = -0.3 * (1.0 - progress * 2.0)
            else:
                vz = 0.0
            
            vx = 0.0
            vy = 0.0
            yaw_rate = 0.0
        
        # Set velocities and integrate
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Safety clamp: enforce height envelope
        self.root_pos[2] = np.clip(self.root_pos[2], 0.15, 0.65)
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        All legs move synchronously. Trajectories designed to maintain
        ground contact during stance and provide appropriate aerial clearance.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Phase 1: Compression [0.0, 0.2]
        if phase < 0.2:
            progress = phase / 0.2
            # Smooth compression: feet rise in body frame as legs shorten
            # Use cosine for smooth start from zero velocity
            compression = self.compression_depth * (1.0 - np.cos(progress * np.pi)) * 0.5
            foot[2] += compression
            
        # Phase 2: Takeoff [0.2, 0.4]
        elif phase < 0.4:
            progress = (phase - 0.2) / 0.2
            # Legs extend rapidly, then lift off
            # First half: extension, second half: liftoff
            if progress < 0.5:
                # Extension phase: feet push down
                extension_progress = progress * 2.0
                extension = self.compression_depth * (1.0 - extension_progress)
                foot[2] += extension
            else:
                # Liftoff phase: feet begin to rise
                liftoff_progress = (progress - 0.5) * 2.0
                lift = self.compression_depth * 0.3 * liftoff_progress
                foot[2] += lift
            
        # Phase 3: Aerial twist [0.4, 0.7]
        elif phase < 0.7:
            progress = (phase - 0.4) / 0.3
            # Legs tuck toward body center to reduce moment of inertia
            tuck_profile = np.sin(progress * np.pi)
            tuck_amount = self.tuck_factor * tuck_profile
            
            # Retract horizontally toward body center
            foot[0] *= (1.0 - tuck_amount * 0.5)
            foot[1] *= (1.0 - tuck_amount * 0.5)
            
            # Lift feet upward
            foot[2] += self.compression_depth * 0.4 + tuck_amount * 0.12
            
        # Phase 4: Landing preparation [0.7, 0.9]
        elif phase < 0.9:
            progress = (phase - 0.7) / 0.2
            # Smooth transition from tucked to extended
            # Inverse tuck profile
            untuck_profile = np.cos(progress * np.pi * 0.5)
            tuck_remaining = self.tuck_factor * untuck_profile
            
            # Return to nominal horizontal positions
            foot[0] = base_pos[0] * (1.0 - tuck_remaining * 0.5) + base_pos[0] * (1.0 - (1.0 - tuck_remaining * 0.5))
            foot[1] = base_pos[1] * (1.0 - tuck_remaining * 0.5) + base_pos[1] * (1.0 - (1.0 - tuck_remaining * 0.5))
            
            # Lower feet toward landing position
            aerial_lift = self.compression_depth * 0.4 + tuck_remaining * 0.12
            foot[2] += aerial_lift * (1.0 - progress)
            
        # Phase 5: Landing [0.9, 1.0]
        else:
            progress = (phase - 0.9) / 0.1
            # Legs compress to absorb impact
            # Feet rise in body frame as legs yield
            # Reduced compression magnitude to prevent penetration
            impact_compression = self.compression_depth * 0.4 * np.sin(progress * np.pi * 0.5)
            foot[2] += impact_compression
        
        return foot