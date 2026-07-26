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
      [0.0, 0.2]: Compression - all legs bend, base lowers
      [0.2, 0.4]: Extension/Liftoff - explosive upward thrust
      [0.4, 0.6]: Aerial peak - all feet off ground, legs tucked
      [0.6, 0.8]: Descent preparation - legs extend for landing
      [0.8, 1.0]: Landing absorption - contact reestablished, compression begins
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0  # 1 Hz bounce frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # All legs synchronized (phase offset = 0 for all)
        self.phase_offsets = {leg: 0.0 for leg in leg_names}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Forward velocity (constant)
        self.vx_forward = 0.5  # m/s forward speed
        
        # Vertical motion parameters
        self.vz_extension_peak = 1.5  # Peak upward velocity during extension
        self.vz_descent_peak = -1.2   # Peak downward velocity during descent
        
        # Leg motion parameters
        self.compression_height = 0.08  # How much legs compress (z offset)
        self.tuck_height = 0.12         # How much legs tuck during aerial phase
        self.tuck_forward = 0.03        # Slight forward drift during tuck
        self.landing_extension = 0.05   # Extra extension for landing preparation
        
        # Horizontal stride (rearward foot drift during stance in body frame)
        self.stance_rearward_drift = 0.04

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
            # Compression: downward velocity decreasing to zero
            progress = phase / 0.2
            vz = self.vz_descent_peak * 0.3 * (1.0 - progress)
            
        elif phase < 0.4:
            # Extension/Liftoff: explosive upward acceleration
            progress = (phase - 0.2) / 0.2
            # Smooth ramp up using sinusoidal ease
            vz = self.vz_extension_peak * np.sin(progress * np.pi / 2)
            
        elif phase < 0.6:
            # Aerial peak: transition from upward to downward
            progress = (phase - 0.4) / 0.2
            # Cosine transition through zero (ballistic arc)
            vz = self.vz_extension_peak * np.cos(progress * np.pi) * 0.5
            
        elif phase < 0.8:
            # Descent: increasing downward velocity
            progress = (phase - 0.6) / 0.2
            vz = self.vz_descent_peak * np.sin(progress * np.pi / 2)
            
        else:
            # Landing absorption: downward velocity rapidly decreasing
            progress = (phase - 0.8) / 0.2
            vz = self.vz_descent_peak * (1.0 - progress)
        
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
        """
        # Get leg-specific phase (all zero offset, so same as global phase)
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine lateral offset sign for forward drift compensation
        # Front legs: positive x, rear legs: negative x in base position
        is_front = leg_name.startswith('F')
        
        if leg_phase < 0.2:
            # Compression phase: foot moves up and back in body frame
            progress = leg_phase / 0.2
            # Smooth compression using sine
            compression_amount = self.compression_height * np.sin(progress * np.pi / 2)
            foot[2] += compression_amount
            # Rearward drift as base moves forward over stationary foot
            foot[0] -= self.stance_rearward_drift * progress
            
        elif leg_phase < 0.4:
            # Extension/Liftoff: foot pushes down then releases
            progress = (leg_phase - 0.2) / 0.2
            # Start compressed, extend back to nominal
            compression_amount = self.compression_height * (1.0 - progress)
            foot[2] += compression_amount
            # Continue rearward drift through early extension
            foot[0] -= self.stance_rearward_drift * (1.0 + progress * 0.5)
            
        elif leg_phase < 0.6:
            # Aerial peak: legs tucked, slight forward positioning
            progress = (leg_phase - 0.4) / 0.2
            # Maximum tuck at mid-aerial phase
            tuck_amount = self.tuck_height * np.sin(progress * np.pi)
            foot[2] += tuck_amount
            # Forward drift to maintain body-relative position during forward travel
            foot[0] += self.tuck_forward * progress
            
        elif leg_phase < 0.8:
            # Descent preparation: extend legs downward for landing
            progress = (leg_phase - 0.6) / 0.2
            # Smooth extension from tucked to extended position
            tuck_reduction = self.tuck_height * np.sin((1.0 - progress) * np.pi)
            extension = self.landing_extension * progress
            foot[2] += tuck_reduction - extension
            # Slight forward positioning anticipating landing
            foot[0] += self.tuck_forward * (1.0 - progress * 0.5)
            
        else:
            # Landing absorption: foot contacts and begins compression
            progress = (leg_phase - 0.8) / 0.2
            # Transition from extended to compressed
            extension_remaining = self.landing_extension * (1.0 - progress)
            compression_starting = self.compression_height * progress * 0.5
            foot[2] += compression_starting - extension_remaining
            # Begin rearward drift as landing occurs
            foot[0] -= self.stance_rearward_drift * progress * 0.3
        
        return foot