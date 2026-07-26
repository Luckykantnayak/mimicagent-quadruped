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
      0.70-0.85: Landing preparation (reduce pitch rate, extend legs)
      0.85-1.00: Landing and stabilization (re-establish contact, zero velocities)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Scaled down motion parameters to stay within height and joint limits
        self.crouch_depth = 0.04          # Reduced from 0.10 to avoid deep knee flexion
        self.launch_vz = 1.3              # Reduced from 3.5 to limit height gain
        self.launch_pitch_rate = 9.0      # Reduced from 12.0 for smoother rotation
        self.flight_pitch_rate = 8.5      # Reduced from 10.0 for smoother rotation
        self.tuck_retraction = 0.07       # Reduced from 0.15 to keep within joint workspace
        
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
        Scaled down to keep base height within safe envelope.
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
        
        # Phase 0.30-0.70: Airborne rotation (shortened duration)
        elif phase < 0.70:
            local_phase = (phase - 0.30) / 0.40
            # Early peak, then descend: steeper parabola
            vx = 0.08
            vz = self.launch_vz * 0.6 * (1.0 - 3.0 * local_phase)
            pitch_rate = self.flight_pitch_rate
        
        # Phase 0.70-0.85: Landing preparation
        elif phase < 0.85:
            local_phase = (phase - 0.70) / 0.15
            # Descending, smoothly reduce pitch rate
            vx = 0.0
            vz = -1.0 * (1.0 - 0.3 * local_phase)
            pitch_rate = self.flight_pitch_rate * (1.0 - local_phase)
        
        # Phase 0.85-1.00: Landing and stabilization
        else:
            local_phase = (phase - 0.85) / 0.15
            # Smooth velocity decay to zero
            vx = 0.0
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
        All legs move synchronously through crouch, tuck, and landing.
        Conservative retraction to stay within joint limits.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
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
            # Smooth tuck: primarily vertical retraction (z direction)
            # Less aggressive horizontal retraction to avoid extreme joint angles
            tuck_envelope = np.sin(np.pi * min(local_phase * 1.8, 1.0))
            
            # Primarily lift feet upward in body frame (reduce z magnitude)
            foot[2] += self.tuck_retraction * tuck_envelope
            
            # Gentle inward retraction in horizontal plane
            foot[0] *= (1.0 - 0.5 * self.tuck_retraction * tuck_envelope)
            foot[1] *= (1.0 - 0.5 * self.tuck_retraction * tuck_envelope)
        
        # Phase 0.70-0.85: Landing preparation (extend legs downward)
        elif phase < 0.85:
            local_phase = (phase - 0.70) / 0.15
            # Smooth transition from tucked to extended
            tuck_factor = 1.0 - local_phase
            tuck_envelope = np.sin(np.pi * min(tuck_factor * 1.8, 1.0))
            
            # Reverse the tuck
            foot[2] += self.tuck_retraction * tuck_envelope
            foot[0] *= (1.0 - 0.5 * self.tuck_retraction * tuck_envelope)
            foot[1] *= (1.0 - 0.5 * self.tuck_retraction * tuck_envelope)
        
        # Phase 0.85-1.00: Landing and stabilization
        else:
            local_phase = (phase - 0.85) / 0.15
            # Gentle compression on impact using smooth envelope
            compression = 0.03 * np.sin(np.pi * local_phase)
            foot[2] += compression
        
        return foot