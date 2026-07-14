from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_TIDAL_SWEEP_LATERAL_MotionGenerator(BaseMotionGenerator):
    """
    Tidal sweep lateral motion: A continuous sideways locomotion pattern where
    a coordinated 'wave' of lateral leg sweeps propagates from front to rear legs,
    producing smooth leftward translation through alternating outward and inward
    leg motions while maintaining continuous ground contact.
    
    Phase structure:
      [0.0, 0.3]: Front legs sweep outward (leftward), rear legs stable
      [0.3, 0.6]: Wave propagates, all legs engaged, peak lateral velocity
      [0.6, 0.9]: Rear legs complete sweep, front legs return inward
      [0.9, 1.0]: All legs reset to neutral stance
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.  # Slower frequency for smooth wave propagation
        
        # Base foot positions (neutral stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Lateral sweep parameters
        self.sweep_amplitude = 0.18  # Maximum lateral displacement (meters)
        
        # Phase offsets for wave propagation (front-to-rear)
        # Front legs lead, rear legs delayed by ~0.3 phase
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('F'):  # Front legs
                self.phase_offsets[leg] = 0.0
            elif leg.startswith('R'):  # Rear legs
                self.phase_offsets[leg] = 0.3
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Lateral velocity parameters (leftward = negative y in body frame)
        self.vy_peak = -0.3  # Peak leftward velocity (m/s)

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent lateral velocity profile.
        
        Velocity profile:
          [0.0, 0.3]: Moderate leftward velocity (initiation)
          [0.3, 0.6]: Peak leftward velocity (full wave engagement)
          [0.6, 0.9]: Decreasing leftward velocity (completion)
          [0.9, 1.0]: Near-zero velocity (reset/settle)
        """
        
        # Compute lateral velocity based on phase
        if phase < 0.3:
            # Front sweep initiation: ramp up to moderate velocity
            progress = phase / 0.3
            vy = self.vy_peak * 0.6 * progress
        elif phase < 0.6:
            # Wave propagation: peak velocity with smooth envelope
            mid_phase = (phase - 0.3) / 0.3
            vy = self.vy_peak * (1.0 - 0.15 * np.sin(np.pi * mid_phase))
        elif phase < 0.9:
            # Rear sweep completion: decelerate smoothly
            progress = (phase - 0.6) / 0.3
            vy = self.vy_peak * (1.0 - progress) * 0.7
        else:
            # Reset and settle: coast to near-zero
            progress = (phase - 0.9) / 0.1
            vy = self.vy_peak * 0.3 * (1.0 - progress)
        
        # Set velocity commands (world frame)
        self.vel_world = np.array([0.0, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
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
        Compute foot position in body frame using smooth lateral sweeping trajectory.
        
        All feet remain in ground contact (z = base_z) throughout entire cycle.
        Lateral (y) displacement follows sinusoidal wave pattern with phase offset.
        
        Front legs (FL, FR):
          [0.0, 0.3]: Sweep outward (y decreases, leftward)
          [0.3, 0.6]: Hold extended position
          [0.6, 0.9]: Return inward (y increases, rightward)
          [0.9, 1.0]: Settle at neutral
        
        Rear legs (RL, RR):
          [0.0, 0.3]: Minimal motion (stable support)
          [0.3, 0.6]: Sweep outward (delayed wave)
          [0.6, 0.9]: Complete sweep and begin return
          [0.9, 1.0]: Rapid return to neutral
        """
        
        # Get base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute leg-specific phase with offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Lateral displacement using smooth trajectory function
        lateral_offset = self._compute_lateral_offset(leg_phase, leg_name)
        
        # Apply lateral displacement (y-axis in body frame, negative = leftward)
        foot[1] += lateral_offset
        
        # Z remains constant (ground contact maintained)
        # foot[2] unchanged from base position
        
        return foot

    def _compute_lateral_offset(self, leg_phase, leg_name):
        """
        Compute smooth lateral offset trajectory for a given leg phase.
        
        Uses sinusoidal blending for C1 continuity across phase boundaries.
        
        Returns:
            lateral_offset: displacement in body y-axis (negative = leftward sweep)
        """
        
        is_front_leg = leg_name.startswith('F')
        
        if is_front_leg:
            # Front legs: early sweep, hold, return, settle
            if leg_phase < 0.3:
                # Outward sweep: 0 -> -sweep_amplitude
                progress = leg_phase / 0.3
                lateral_offset = -self.sweep_amplitude * (1.0 - np.cos(np.pi * progress)) / 2.0
            elif leg_phase < 0.6:
                # Hold extended: maintain near -sweep_amplitude
                lateral_offset = -self.sweep_amplitude
            elif leg_phase < 0.9:
                # Inward return: -sweep_amplitude -> 0
                progress = (leg_phase - 0.6) / 0.3
                lateral_offset = -self.sweep_amplitude * (1.0 + np.cos(np.pi * progress)) / 2.0
            else:
                # Settle: approach 0 smoothly
                progress = (leg_phase - 0.9) / 0.1
                lateral_offset = -self.sweep_amplitude * 0.05 * (1.0 - progress)
        
        else:
            # Rear legs: stable, delayed sweep, complete and return, rapid reset
            if leg_phase < 0.3:
                # Minimal motion: stable support with slight preparation
                lateral_offset = 0.0
            elif leg_phase < 0.6:
                # Outward sweep: 0 -> -sweep_amplitude (delayed wave)
                progress = (leg_phase - 0.3) / 0.3
                lateral_offset = -self.sweep_amplitude * (1.0 - np.cos(np.pi * progress)) / 2.0
            elif leg_phase < 0.9:
                # Complete sweep and begin return: -sweep_amplitude -> -0.3*sweep_amplitude
                progress = (leg_phase - 0.6) / 0.3
                mid_progress = np.sin(np.pi * progress)
                # Peak at progress=0, then return partially
                lateral_offset = -self.sweep_amplitude * (1.0 - 0.7 * progress)
            else:
                # Rapid return to neutral: -0.3*sweep_amplitude -> 0
                progress = (leg_phase - 0.9) / 0.1
                lateral_offset = -self.sweep_amplitude * 0.3 * (1.0 - progress)
        
        return lateral_offset