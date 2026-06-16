from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_POGO_BOUNCE_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Pogo-stick bouncing gait with synchronized leg motion.
    
    All four legs compress and extend in unison, creating rhythmic vertical
    oscillation with aerial phases. The base maintains constant forward
    velocity while vertical velocity pulses through the bounce cycle.
    
    Phase breakdown:
      [0.0, 0.2]: Compression - legs slightly compressed, preparing for extension
      [0.2, 0.4]: Extension/Liftoff - explosive upward thrust, legs extend then lift
      [0.4, 0.6]: Aerial peak - all feet off ground, legs tucked
      [0.6, 0.8]: Descent preparation - legs extend downward for landing
      [0.8, 1.0]: Landing absorption - contact reestablished, legs compress
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0  # 1 Hz bounce frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # All legs synchronized (phase offset = 0 for all)
        self.phase_offsets = {leg: 0.0 for leg in leg_names}
        
        # Base state - initialize at proper standing height
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, 0.30])  # Start at nominal standing height
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Forward velocity (constant)
        self.vx_forward = 0.5  # m/s forward speed
        
        # Vertical motion parameters - balanced for zero net drift per cycle
        self.vz_extension_peak = 1.15  # Peak upward velocity during extension
        self.vz_descent_peak = -0.7    # Peak downward velocity during descent
        
        # Leg motion parameters (small values to avoid over-extension)
        self.compression_retraction = 0.025  # Feet slightly closer to body during compression
        self.tuck_retraction = 0.065        # Feet retract during aerial phase
        self.landing_extension = 0.020      # Slight downward extension for landing
        
        # Horizontal stride (rearward foot drift during stance in body frame)
        self.stance_rearward_drift = 0.025

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Energy-balanced vertical velocity profile to prevent base drift.
        """
        vx = self.vx_forward
        vy = 0.0
        
        # Vertical velocity profile - designed for zero net displacement per cycle
        if phase < 0.2:
            # Compression: slight upward velocity ramp (preparing for extension)
            progress = phase / 0.2
            vz = 0.05 * progress
            
        elif phase < 0.4:
            # Extension/Liftoff: explosive upward acceleration
            progress = (phase - 0.2) / 0.2
            vz = self.vz_extension_peak * np.sin(progress * np.pi / 2)
            
        elif phase < 0.6:
            # Aerial peak: symmetric ballistic transition through apex
            progress = (phase - 0.4) / 0.2
            vz = (self.vz_extension_peak * 0.5) * np.cos(progress * np.pi)
            
        elif phase < 0.8:
            # Descent: downward velocity increases
            progress = (phase - 0.6) / 0.2
            vz = self.vz_descent_peak * np.sin(progress * np.pi / 2)
            
        else:
            # Landing absorption: rapid downward velocity decrease
            progress = (phase - 0.8) / 0.2
            vz = self.vz_descent_peak * (1.0 - progress ** 1.5)
        
        # Set velocity commands (world frame)
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, 0.0])  # No rotation
        
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
        Compute foot position in body frame for given phase.
        
        All legs move identically (synchronized pogo bounce).
        Convention: 
        - Positive z-offset: retract feet upward (closer to body)
        - Negative z-offset: extend feet downward (farther from body)
        """
        # Get leg-specific phase (all zero offset, so same as global phase)
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_phase < 0.2:
            # Compression: legs compressed, feet slightly closer to body
            progress = leg_phase / 0.2
            # Slight retraction with smooth profile
            z_offset = self.compression_retraction * (1.0 - 0.3 * progress)
            foot[2] += z_offset  # Retract upward
            # Rearward drift as base moves forward
            foot[0] -= self.stance_rearward_drift * progress
            
        elif leg_phase < 0.4:
            # Extension/Liftoff: legs extend then begin to lift
            progress = (leg_phase - 0.2) / 0.2
            # Smooth transition from compressed to nominal to slightly retracted
            if progress < 0.6:
                # Extend from compressed toward nominal
                z_offset = self.compression_retraction * (1.0 - progress / 0.6) * 0.7
                foot[2] += z_offset
            else:
                # Begin slight retraction as liftoff occurs
                lift_progress = (progress - 0.6) / 0.4
                z_offset = 0.015 * lift_progress
                foot[2] += z_offset
            # Continue rearward drift
            foot[0] -= self.stance_rearward_drift * (1.0 + progress * 0.8)
            
        elif leg_phase < 0.6:
            # Aerial peak: legs tucked (maximum retraction)
            progress = (leg_phase - 0.4) / 0.2
            # Smooth tuck profile with peak at mid-phase
            tuck_amount = self.tuck_retraction * (0.3 + 0.7 * np.sin((progress - 0.5) * np.pi + np.pi / 2))
            foot[2] += tuck_amount  # Retract upward
            # Slight forward drift to maintain body-relative position
            foot[0] += 0.015 * np.sin(progress * np.pi)
            
        elif leg_phase < 0.8:
            # Descent preparation: legs extend downward for landing
            progress = (leg_phase - 0.6) / 0.2
            # Smooth transition from tucked to extended
            tuck_remaining = self.tuck_retraction * (1.0 - progress)
            landing_ext = self.landing_extension * progress
            z_offset = tuck_remaining - landing_ext
            foot[2] += z_offset  # Net: from retracted to slightly extended
            
        else:
            # Landing absorption: contact established, begin compressing
            progress = (leg_phase - 0.8) / 0.2
            # Smooth transition from extended to compressed
            ext_remaining = self.landing_extension * (1.0 - progress)
            compression_start = self.compression_retraction * (progress ** 0.8)
            z_offset = compression_start - ext_remaining
            foot[2] += z_offset  # Smooth continuous transition
            # Begin rearward drift
            foot[0] -= self.stance_rearward_drift * progress * 0.4
        
        return foot