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
        # Calculate appropriate base height so feet start at ground
        avg_foot_z = np.mean([v[2] for v in self.base_feet_pos_body.values()])
        self.initial_base_height = -avg_foot_z + 0.03
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.initial_base_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Kinematic parameters
        self.compression_depth = 0.08  # Conservative compression
        self.tuck_factor = 0.35  # Moderate tuck for stability
        
        # Velocity parameters tuned to stay within envelope
        self.takeoff_vx = 1.0  # Forward velocity (m/s)
        self.takeoff_vz = 1.0  # Upward velocity (m/s)
        
        # Yaw rotation parameters (180 degrees over 0.3 phase duration)
        self.yaw_rate_aerial = np.pi / (0.3 / self.freq)  # rad/s
        
        # Target landing height (base height when feet should contact ground)
        self.landing_base_height = self.initial_base_height
        
    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on current phase.
        Carefully tuned to prevent ground penetration and maintain envelope.
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
            progress = phase / 0.2
            # Gentle downward motion
            vz = -0.3 * np.sin(progress * np.pi)
            
        # Phase 2: Takeoff [0.2, 0.4]
        elif phase < 0.4:
            # Explosive but controlled launch
            progress = (phase - 0.2) / 0.2
            accel_profile = np.sin(progress * np.pi)
            vx = self.takeoff_vx * accel_profile
            vz = self.takeoff_vz * accel_profile
            
        # Phase 3: Aerial twist [0.4, 0.7]
        elif phase < 0.7:
            # Ballistic trajectory with yaw rotation
            progress = (phase - 0.4) / 0.3
            
            # Maintain forward velocity
            vx = self.takeoff_vx * 0.8
            
            # Parabolic vertical trajectory
            apex_progress = 0.5
            if progress < apex_progress:
                # Ascending to apex
                vz = self.takeoff_vz * 0.35 * (1.0 - progress / apex_progress)
            else:
                # Descending from apex
                descent_progress = (progress - apex_progress) / (1.0 - apex_progress)
                vz = -0.5 * descent_progress
            
            # Constant yaw rotation
            yaw_rate = self.yaw_rate_aerial
            
        # Phase 4: Landing preparation [0.7, 0.9]
        elif phase < 0.9:
            # CRITICAL FIX: Controlled descent at much lower velocity
            progress = (phase - 0.7) / 0.2
            
            # Decelerate forward motion
            vx = self.takeoff_vx * 0.5 * (1.0 - progress)
            
            # Gentle controlled descent (reduced from -3.4~-5.1 to -0.6~-0.8 range)
            vz = -0.65 * (1.0 + progress * 0.2)
            
            yaw_rate = 0.0
            
        # Phase 5: Landing [0.9, 1.0]
        else:
            # CRITICAL FIX: Zero or slight upward base velocity to maintain ground contact
            progress = (phase - 0.9) / 0.1
            
            # No downward motion - hold position or slight upward compensation
            vz = 0.15 * (1.0 - progress)  # Gentle upward motion to counteract leg compression
            
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
        self.root_pos[2] = np.clip(self.root_pos[2], 0.18, 0.62)
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        All legs move synchronously. Landing phase redesigned to prevent penetration.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Phase 1: Compression [0.0, 0.2]
        if phase < 0.2:
            progress = phase / 0.2
            # Smooth compression: feet rise in body frame as legs compress
            compression = self.compression_depth * (1.0 - np.cos(progress * np.pi)) * 0.5
            foot[2] += compression
            
        # Phase 2: Takeoff [0.2, 0.4]
        elif phase < 0.4:
            progress = (phase - 0.2) / 0.2
            # Legs extend then lift off
            if progress < 0.6:
                # Extension phase
                extension_progress = progress / 0.6
                extension = self.compression_depth * (1.0 - extension_progress)
                foot[2] += extension
            else:
                # Liftoff phase
                liftoff_progress = (progress - 0.6) / 0.4
                lift = self.compression_depth * 0.25 * liftoff_progress
                foot[2] += lift
            
        # Phase 3: Aerial twist [0.4, 0.7]
        elif phase < 0.7:
            progress = (phase - 0.4) / 0.3
            # Tuck legs toward body center
            tuck_profile = np.sin(progress * np.pi)
            tuck_amount = self.tuck_factor * tuck_profile
            
            # Retract horizontally
            foot[0] *= (1.0 - tuck_amount * 0.45)
            foot[1] *= (1.0 - tuck_amount * 0.45)
            
            # Lift feet upward
            foot[2] += self.compression_depth * 0.35 + tuck_amount * 0.1
            
        # Phase 4: Landing preparation [0.7, 0.9]
        elif phase < 0.9:
            progress = (phase - 0.7) / 0.2
            # Smooth transition from tucked to fully extended
            untuck_profile = np.cos(progress * np.pi * 0.5)
            tuck_remaining = self.tuck_factor * untuck_profile
            
            # Interpolate back to nominal stance positions
            current_retract = 1.0 - tuck_remaining * 0.45
            foot[0] = base_pos[0] * current_retract
            foot[1] = base_pos[1] * current_retract
            
            # Lower feet toward landing position - ensure full extension by end
            aerial_lift = self.compression_depth * 0.35 + tuck_remaining * 0.1
            foot[2] += aerial_lift * (1.0 - progress)
            
            # Add slight downward bias at end to reach for ground
            if progress > 0.7:
                foot[2] -= 0.01 * ((progress - 0.7) / 0.3)
            
        # Phase 5: Landing [0.9, 1.0]
        else:
            # CRITICAL FIX: Feet should NOT rise during landing
            # Instead, maintain nominal stance position or extend slightly downward
            progress = (phase - 0.9) / 0.1
            
            # Option 1: Keep feet at nominal position (remove upward compression)
            # foot[2] remains at base_pos[2]
            
            # Option 2: Slight downward extension to ensure ground contact
            foot[2] -= 0.02 * np.sin(progress * np.pi * 0.5)
        
        return foot