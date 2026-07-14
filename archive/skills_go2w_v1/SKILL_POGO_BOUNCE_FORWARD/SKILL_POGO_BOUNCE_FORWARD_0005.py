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
      [0.0, 0.2]: Compression - all legs extend downward, base lowers
      [0.2, 0.4]: Extension/Liftoff - legs retract, base rises
      [0.4, 0.6]: Aerial peak - all feet off ground, legs tucked
      [0.6, 0.8]: Descent preparation - legs extend downward for landing
      [0.8, 1.0]: Landing absorption - legs extend to contact, begin retracting
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
        
        # Vertical motion parameters (reduced magnitude to prevent excessive oscillation)
        self.vz_extension_peak = 1.0  # Peak upward velocity during extension
        self.vz_descent_peak = -0.7   # Peak downward velocity during descent
        
        # Leg motion parameters (reinterpreted: positive = extend down, negative = retract up)
        self.stance_extension = 0.05  # How much legs extend downward during stance
        self.tuck_retraction = 0.10   # How much legs retract upward during aerial phase
        self.landing_extension = 0.08 # Downward extension for landing preparation
        
        # Horizontal stride (rearward foot drift during stance in body frame)
        self.stance_rearward_drift = 0.03

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
            # Compression: slight downward velocity decreasing to zero
            # This is the end of landing absorption transitioning to push-off
            progress = phase / 0.2
            vz = -0.2 * (1.0 - progress)  # Start slightly negative, approach zero
            
        elif phase < 0.4:
            # Extension/Liftoff: explosive upward acceleration
            progress = (phase - 0.2) / 0.2
            # Smooth ramp up using sinusoidal ease
            vz = self.vz_extension_peak * np.sin(progress * np.pi / 2)
            
        elif phase < 0.6:
            # Aerial peak: transition from upward to downward (ballistic)
            progress = (phase - 0.4) / 0.2
            # Cosine transition through zero
            vz = self.vz_extension_peak * np.cos(progress * np.pi / 2) - 0.3 * progress
            
        elif phase < 0.8:
            # Descent: increasing downward velocity
            progress = (phase - 0.6) / 0.2
            vz = self.vz_descent_peak * np.sin(progress * np.pi / 2)
            
        else:
            # Landing absorption: downward velocity rapidly decreasing
            progress = (phase - 0.8) / 0.2
            vz = self.vz_descent_peak * (1.0 - progress * 0.7)
        
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
        Key: Positive z_offset = extend DOWN (away from body)
             Negative z_offset = retract UP (toward body)
        """
        # Get leg-specific phase (all zero offset, so same as global phase)
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if front or rear leg for forward drift compensation
        is_front = leg_name.startswith('F')
        
        if leg_phase < 0.2:
            # Compression/Preparation: legs slightly extended, preparing for push-off
            # Feet stay in contact with ground, extended below body
            progress = leg_phase / 0.2
            # Maintain extension with slight increase preparing for liftoff
            z_offset = self.stance_extension * (0.8 + 0.2 * progress)
            foot[2] -= z_offset  # Extend downward (negative in body frame z)
            # Rearward drift as base moves forward over stationary foot
            foot[0] -= self.stance_rearward_drift * progress
            
        elif leg_phase < 0.4:
            # Extension/Liftoff: legs retract as base accelerates upward
            # Transition from extended to retracted as feet leave ground
            progress = (leg_phase - 0.2) / 0.2
            # Smooth transition from extended to neutral then tucked
            extension_amount = self.stance_extension * (1.0 - progress)
            retraction_start = self.tuck_retraction * 0.3 * progress
            z_offset = extension_amount - retraction_start
            foot[2] -= z_offset
            # Continue rearward drift during early liftoff
            foot[0] -= self.stance_rearward_drift * (1.0 + progress * 0.5)
            
        elif leg_phase < 0.6:
            # Aerial peak: legs fully tucked (retracted upward toward body)
            progress = (leg_phase - 0.4) / 0.2
            # Maximum tuck in middle of aerial phase
            tuck_progress = np.sin(progress * np.pi)
            z_offset = -self.tuck_retraction * (0.3 + 0.7 * tuck_progress)
            foot[2] -= z_offset  # Retract upward (positive offset, negative application)
            # Slight forward drift to maintain body-relative position
            foot[0] += 0.02 * progress
            
        elif leg_phase < 0.8:
            # Descent preparation: legs extend downward aggressively for landing
            progress = (leg_phase - 0.6) / 0.2
            # Transition from tucked to fully extended below body
            tuck_remaining = self.tuck_retraction * (1.0 - progress)
            landing_extension = self.landing_extension * progress
            z_offset = -tuck_remaining + landing_extension
            foot[2] -= z_offset  # Net downward extension
            # Prepare for contact with slight forward positioning
            foot[0] += 0.02 * (1.0 - progress)
            
        else:
            # Landing absorption: legs extended at contact, beginning to retract
            # This phase shows initial contact and start of compression cycle
            progress = (leg_phase - 0.8) / 0.2
            # Start fully extended, gradually reduce extension
            z_offset = self.landing_extension * (1.0 - 0.4 * progress)
            foot[2] -= z_offset  # Maintain downward extension during absorption
            # Begin rearward drift as landing establishes contact
            foot[0] -= self.stance_rearward_drift * progress * 0.5
        
        return foot