import numpy as np
from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_FRONT_FLIP_MotionGenerator(BaseMotionGenerator):
    """
    Front flip motion: Complete forward pitch rotation (360°) with airborne phase.
    
    Phase breakdown:
      0.00-0.15: Crouch preparation (all feet grounded, body lowers)
      0.15-0.30: Launch and rotation initiation (explosive upward + pitch rate)
      0.30-0.65: Airborne rotation (all feet off ground, complete ~360° pitch)
      0.65-0.80: Pitch decay & tuck release (feet return to neutral, body uprights)
      0.80-0.90: Gentle landing extension (minimal extension, body nearly upright)
      0.90-1.00: Landing and stabilization (re-establish contact, zero velocities)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - hybrid approach to avoid both joint limits and ground penetration
        self.crouch_depth = 0.06          # Moderate crouch for flexion range
        self.launch_vz = 1.3              # Controlled upward velocity
        self.launch_pitch_rate = 9.0      # Smooth rotation initiation
        self.flight_pitch_rate = 8.5      # Sustained rotation rate
        self.tuck_retraction = 0.07       # Conservative tuck within joint workspace
        self.landing_extension = 0.025    # Minimal extension applied only when body is upright
        
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
        Extended pitch decay window to ensure body is upright before landing extension.
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
            envelope = np.sin(np.pi * local_phase)
            vx = 0.15 * envelope
            vz = self.launch_vz * envelope
            pitch_rate = self.launch_pitch_rate * (0.3 + 0.7 * local_phase)
        
        # Phase 0.30-0.65: Airborne rotation (extended duration)
        elif phase < 0.65:
            local_phase = (phase - 0.30) / 0.35
            vx = 0.08
            vz = self.launch_vz * 0.6 * (1.0 - 3.0 * local_phase)
            pitch_rate = self.flight_pitch_rate
        
        # Phase 0.65-0.80: Early landing prep - pitch decay, gentle descent
        elif phase < 0.80:
            local_phase = (phase - 0.65) / 0.15
            vx = 0.0
            # Gentle descent to begin approaching ground
            vz = -0.6 * local_phase
            # Extended pitch rate decay with quadratic profile for smooth uprighting
            pitch_rate = self.flight_pitch_rate * (1.0 - local_phase)**2.0
        
        # Phase 0.80-0.90: Late landing prep - body nearly upright, continued descent
        elif phase < 0.90:
            local_phase = (phase - 0.80) / 0.10
            vx = 0.0
            # Continued descent
            vz = -0.6 - 0.3 * local_phase
            # Pitch rate nearly zero
            pitch_rate = 0.0
        
        # Phase 0.90-1.00: Landing and stabilization
        else:
            local_phase = (phase - 0.90) / 0.10
            vx = 0.0
            # Smooth velocity decay to zero
            vz = -0.3 * (1.0 - local_phase)
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
        Key fix: Landing extension only applied AFTER body is upright (phase > 0.80).
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Phase 0.00-0.15: Crouch preparation
        if phase < 0.15:
            local_phase = phase / 0.15
            crouch_amount = self.crouch_depth * (1.0 - np.cos(np.pi * local_phase)) * 0.5
            foot[2] += crouch_amount
        
        # Phase 0.15-0.30: Launch (feet push off, then break contact)
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            crouch_amount = self.crouch_depth * (1.0 - np.cos(np.pi * (1.0 - local_phase))) * 0.5
            foot[2] += crouch_amount
        
        # Phase 0.30-0.65: Airborne rotation (tuck feet toward body)
        elif phase < 0.65:
            local_phase = (phase - 0.30) / 0.35
            tuck_envelope = np.sin(np.pi * min(local_phase * 1.8, 1.0))
            
            # Lift feet upward in body frame
            foot[2] += self.tuck_retraction * tuck_envelope
            
            # Gentle inward retraction in horizontal plane
            foot[0] *= (1.0 - 0.5 * self.tuck_retraction * tuck_envelope)
            foot[1] *= (1.0 - 0.5 * self.tuck_retraction * tuck_envelope)
        
        # Phase 0.65-0.80: Tuck release ONLY - return to neutral stance (no extension yet)
        elif phase < 0.80:
            local_phase = (phase - 0.65) / 0.15
            
            # Gradually release tuck, returning feet to base positions
            tuck_release = 1.0 - local_phase
            remaining_tuck = self.tuck_retraction * np.sin(np.pi * min(tuck_release * 1.8, 1.0))
            foot[2] += remaining_tuck
            
            # Release horizontal retraction
            horizontal_release = self.tuck_retraction * np.sin(np.pi * min(tuck_release * 1.8, 1.0))
            foot[0] *= (1.0 - 0.5 * horizontal_release)
            foot[1] *= (1.0 - 0.5 * horizontal_release)
        
        # Phase 0.80-0.90: Minimal landing extension (body is now upright)
        elif phase < 0.90:
            local_phase = (phase - 0.80) / 0.10
            
            # Apply minimal downward extension now that body frame is nearly aligned with world
            extension_amount = self.landing_extension * local_phase
            foot[2] -= extension_amount
        
        # Phase 0.90-1.00: Landing and stabilization
        else:
            local_phase = (phase - 0.90) / 0.10
            
            # Maintain extension in first half, then gently release with small compression
            if local_phase < 0.6:
                # Maintain extension plus small compression on impact
                compression = 0.02 * np.sin(np.pi * local_phase / 0.6)
                foot[2] -= (self.landing_extension + compression)
            else:
                # Release toward neutral
                release_phase = (local_phase - 0.6) / 0.4
                remaining_extension = self.landing_extension * (1.0 - release_phase)
                foot[2] -= remaining_extension
        
        return foot