import numpy as np
from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_FRONT_FLIP_MotionGenerator(BaseMotionGenerator):
    """
    Front flip motion: Complete forward pitch rotation (360°) with airborne phase.
    
    Phase breakdown:
      0.00-0.15: Crouch preparation (all feet grounded, body lowers)
      0.15-0.30: Launch and rotation initiation (explosive upward + pitch rate)
      0.30-0.70: Airborne rotation (all feet off ground, complete ~360° pitch)
      0.70-0.85: Landing preparation (reduce pitch rate, extend legs downward)
      0.85-1.00: Landing and stabilization (re-establish contact, zero velocities)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters tuned to avoid height violations, joint limits, and ground penetration
        self.crouch_depth = 0.06          # Moderate crouch for leg extension range
        self.launch_vz = 1.3              # Controlled upward velocity
        self.launch_pitch_rate = 9.0      # Smooth rotation initiation
        self.flight_pitch_rate = 8.5      # Sustained rotation rate
        self.tuck_retraction = 0.07       # Conservative tuck to stay within joint workspace
        self.landing_extension = 0.08     # Downward leg extension for landing prep
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Controlled descent during landing to coordinate with foot extension.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 0.00-0.15: Crouch preparation
        if phase < 0.15:
            vx = 0.0
            vz = 0.0
            pitch_rate = 0.0
        
        # Phase 0.15-0.30: Launch and rotation initiation
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            # Smooth ramp up with sin envelope
            envelope = np.sin(np.pi * local_phase)
            vx = 0.15 * envelope
            vz = self.launch_vz * envelope
            pitch_rate = self.launch_pitch_rate * (0.3 + 0.7 * local_phase)
        
        # Phase 0.30-0.70: Airborne rotation
        elif phase < 0.70:
            local_phase = (phase - 0.30) / 0.40
            # Early peak, then descend
            vx = 0.08
            vz = self.launch_vz * 0.6 * (1.0 - 3.0 * local_phase)
            pitch_rate = self.flight_pitch_rate
        
        # Phase 0.70-0.85: Landing preparation - reduced descent rate
        elif phase < 0.85:
            local_phase = (phase - 0.70) / 0.15
            # Gentler descent to allow foot extension to catch up
            vx = 0.0
            vz = -0.7 * (1.0 - 0.2 * local_phase)
            # Extended pitch rate decay window for better body alignment
            pitch_rate = self.flight_pitch_rate * (1.0 - local_phase)**1.5
        
        # Phase 0.85-1.00: Landing and stabilization
        else:
            local_phase = (phase - 0.85) / 0.15
            # Smooth velocity decay to zero
            vx = 0.0
            vz = -0.25 * (1.0 - local_phase)
            pitch_rate = 0.0
        
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
        Compute foot position in body frame based on phase.
        CRITICAL FIX: Landing preparation now extends feet DOWNWARD (negative z offset).
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        
        # Phase 0.00-0.15: Crouch preparation
        if phase < 0.15:
            local_phase = phase / 0.15
            # Smooth crouch using cosine ease
            crouch_amount = self.crouch_depth * (1.0 - np.cos(np.pi * local_phase)) * 0.5
            foot[2] += crouch_amount
        
        # Phase 0.15-0.30: Launch (feet push off, then break contact)
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            # Smooth transition from crouch to neutral
            crouch_amount = self.crouch_depth * (1.0 - np.cos(np.pi * (1.0 - local_phase))) * 0.5
            foot[2] += crouch_amount
        
        # Phase 0.30-0.70: Airborne rotation (tuck feet toward body)
        elif phase < 0.70:
            local_phase = (phase - 0.30) / 0.40
            # Smooth tuck: primarily vertical retraction
            tuck_envelope = np.sin(np.pi * min(local_phase * 1.8, 1.0))
            
            # Lift feet upward in body frame (positive z offset)
            foot[2] += self.tuck_retraction * tuck_envelope
            
            # Gentle inward retraction in horizontal plane
            foot[0] *= (1.0 - 0.5 * self.tuck_retraction * tuck_envelope)
            foot[1] *= (1.0 - 0.5 * self.tuck_retraction * tuck_envelope)
        
        # Phase 0.70-0.85: Landing preparation - EXTEND LEGS DOWNWARD
        elif phase < 0.85:
            local_phase = (phase - 0.70) / 0.15
            
            # CRITICAL FIX: Extend feet DOWNWARD (subtract from z) as landing approaches
            # Start from tucked position, transition through neutral, then extend below
            tuck_release = 1.0 - local_phase
            remaining_tuck = self.tuck_retraction * np.sin(np.pi * min(tuck_release * 1.8, 1.0))
            
            # Apply remaining tuck offset (decreases as we approach landing)
            foot[2] += remaining_tuck
            
            # Add downward extension for landing (increases as we approach landing)
            extension_amount = self.landing_extension * local_phase
            foot[2] -= extension_amount  # NEGATIVE offset extends legs down
            
            # Release horizontal retraction - return to full stance width
            horizontal_release = self.tuck_retraction * np.sin(np.pi * min(tuck_release * 1.8, 1.0))
            foot[0] *= (1.0 - 0.5 * horizontal_release)
            foot[1] *= (1.0 - 0.5 * horizontal_release)
            
            # Front legs get slightly more extension to compensate for residual pitch
            if is_front:
                foot[2] -= 0.02 * local_phase
        
        # Phase 0.85-1.00: Landing and stabilization
        else:
            local_phase = (phase - 0.85) / 0.15
            
            # Compression profile: extend down initially, then release
            # Peak compression at mid-phase, return to neutral by end
            if local_phase < 0.5:
                # First half: maintain extension and compress further on impact
                compression = 0.05 * np.sin(np.pi * local_phase * 2.0)
                foot[2] -= (self.landing_extension + compression)
            else:
                # Second half: release compression back to neutral
                release_phase = (local_phase - 0.5) * 2.0
                remaining_extension = self.landing_extension * (1.0 - release_phase)
                foot[2] -= remaining_extension
            
            # Front legs maintain slight extra extension during early landing
            if is_front and local_phase < 0.5:
                foot[2] -= 0.02 * (1.0 - local_phase * 2.0)
        
        return foot