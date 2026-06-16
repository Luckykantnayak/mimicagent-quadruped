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
      [0.0, 0.2]: Compression - legs slightly compressed (feet closer to body)
      [0.2, 0.4]: Extension/Liftoff - legs extend then lift off
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
        
        # Vertical motion parameters (moderate magnitude for controlled oscillation)
        self.vz_extension_peak = 0.9  # Peak upward velocity during extension
        self.vz_descent_peak = -0.7   # Peak downward velocity during descent
        
        # Leg motion parameters (small values to avoid over-extension)
        # Positive offset = retract upward (feet closer to body)
        # Negative offset = extend downward (feet farther from body)
        self.compression_retraction = 0.025  # Feet slightly closer to body during compression
        self.tuck_retraction = 0.065        # Feet retract during aerial phase
        self.landing_extension = 0.020      # Slight downward extension for landing
        
        # Horizontal stride (rearward foot drift during stance in body frame)
        self.stance_rearward_drift = 0.025

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        - vx: constant forward velocity
        - vy: zero (no lateral motion)
        - vz: phase-dependent vertical velocity profile
        - No angular rates (all zero)
        """
        vx = self.vx_forward
        vy = 0.0
        
        # Vertical velocity profile through bounce cycle
        if phase < 0.2:
            # Compression: minimal downward velocity, approaching zero
            progress = phase / 0.2
            vz = -0.15 * (1.0 - progress)
            
        elif phase < 0.4:
            # Extension/Liftoff: upward acceleration
            progress = (phase - 0.2) / 0.2
            vz = self.vz_extension_peak * np.sin(progress * np.pi / 2)
            
        elif phase < 0.6:
            # Aerial peak: ballistic transition from upward to downward
            progress = (phase - 0.4) / 0.2
            vz = self.vz_extension_peak * 0.5 * np.cos(progress * np.pi) - 0.2 * progress
            
        elif phase < 0.8:
            # Descent: downward velocity increases
            progress = (phase - 0.6) / 0.2
            vz = self.vz_descent_peak * np.sin(progress * np.pi / 2)
            
        else:
            # Landing absorption: downward velocity decreases
            progress = (phase - 0.8) / 0.2
            vz = self.vz_descent_peak * (1.0 - progress * 0.8)
        
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
        - base_feet_pos_body already has feet at nominal positions (~-0.28m in z)
        """
        # Get leg-specific phase (all zero offset, so same as global phase)
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_phase < 0.2:
            # Compression: legs compressed, feet slightly closer to body
            progress = leg_phase / 0.2
            # Start with slight retraction, decrease through phase
            z_offset = self.compression_retraction * (1.0 - 0.5 * progress)
            foot[2] += z_offset  # Retract upward
            # Rearward drift as base moves forward
            foot[0] -= self.stance_rearward_drift * progress
            
        elif leg_phase < 0.4:
            # Extension/Liftoff: legs extend then lift off
            progress = (leg_phase - 0.2) / 0.2
            if progress < 0.5:
                # First half: extend from compressed to nominal
                z_offset = self.compression_retraction * (1.0 - 2.0 * progress)
                foot[2] += z_offset
            else:
                # Second half: begin slight retraction as liftoff occurs
                retract = 0.02 * (progress - 0.5) * 2.0
                foot[2] += retract
            # Continue rearward drift
            foot[0] -= self.stance_rearward_drift * (1.0 + progress)
            
        elif leg_phase < 0.6:
            # Aerial peak: legs tucked (retracted upward)
            progress = (leg_phase - 0.4) / 0.2
            # Smooth tuck profile
            tuck_amount = self.tuck_retraction * (0.5 + 0.5 * np.sin((progress - 0.5) * np.pi))
            foot[2] += tuck_amount  # Retract upward
            # Slight forward drift
            foot[0] += 0.015 * np.sin(progress * np.pi)
            
        elif leg_phase < 0.8:
            # Descent preparation: legs extend downward for landing
            progress = (leg_phase - 0.6) / 0.2
            # Transition from tucked to extended
            tuck_remaining = self.tuck_retraction * (1.0 - progress)
            landing_ext = self.landing_extension * progress
            z_offset = tuck_remaining - landing_ext
            foot[2] += z_offset  # Net: from retracted to slightly extended
            
        else:
            # Landing absorption: feet contact, begin compressing
            progress = (leg_phase - 0.8) / 0.2
            # Transition from extended to compressed
            ext_remaining = self.landing_extension * (1.0 - progress)
            compression_start = self.compression_retraction * progress
            z_offset = compression_start - ext_remaining
            foot[2] += z_offset  # Smooth transition
            # Begin rearward drift
            foot[0] -= self.stance_rearward_drift * progress * 0.4
        
        return foot