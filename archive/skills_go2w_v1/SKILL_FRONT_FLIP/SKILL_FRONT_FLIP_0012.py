import numpy as np
from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_FRONT_FLIP_MotionGenerator(BaseMotionGenerator):
    """
    Front flip motion: Complete forward pitch rotation (360°) with airborne phase.
    
    Based on Iteration 2 foundation (best overall performance) with refined base descent.
    
    Phase breakdown:
      0.00-0.15: Crouch preparation (all feet grounded, body lowers)
      0.15-0.30: Launch and rotation initiation (explosive upward + pitch rate)
      0.30-0.75: Airborne rotation (all feet off ground, complete ~360° pitch)
      0.75-0.90: Landing preparation (reduce pitch rate, extend legs)
      0.90-1.00: Landing and stabilization (re-establish contact, zero velocities)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - return to Iteration 2 foundation with refinements
        self.crouch_depth = 0.04          # Iteration 2 value - proven safe for joint limits
        self.launch_vz = 1.3              # Controlled upward velocity
        self.launch_pitch_rate = 9.0      # Smooth rotation initiation
        self.flight_pitch_rate = 9.5      # Slightly increased from iter 5's 8.5, below iter 2's 10.0
        self.tuck_retraction = 0.07       # Conservative tuck within joint workspace
        
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
        Enhanced base descent to eliminate ground penetration without foot extension.
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
        
        # Phase 0.30-0.75: Airborne rotation (restored longer duration from iteration 2)
        elif phase < 0.75:
            local_phase = (phase - 0.30) / 0.45
            vx = 0.08
            vz = self.launch_vz * 0.6 * (1.0 - 2.5 * local_phase)
            pitch_rate = self.flight_pitch_rate
        
        # Phase 0.75-0.90: Landing preparation - enhanced descent (key fix from iteration 2)
        elif phase < 0.90:
            local_phase = (phase - 0.75) / 0.15
            vx = 0.0
            # Increased from -1.0 to -1.15 m/s, sustained longer
            vz = -1.15 * (1.0 - 0.15 * local_phase)
            # Smooth pitch rate decay
            pitch_rate = self.flight_pitch_rate * (1.0 - local_phase)
        
        # Phase 0.90-1.00: Landing and stabilization - continued descent
        else:
            local_phase = (phase - 0.90) / 0.10
            vx = 0.0
            # Increased from -0.3 to -0.45 m/s for continued approach to ground
            vz = -0.45 * (1.0 - local_phase)
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
        Iteration 2 approach: feet return to neutral stance, NO downward extension.
        All foot[2] offsets are ADDITIVE (upward in body frame).
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
        
        # Phase 0.30-0.75: Airborne rotation (tuck feet toward body)
        elif phase < 0.75:
            local_phase = (phase - 0.30) / 0.45
            # Smooth tuck: primarily vertical retraction
            tuck_envelope = np.sin(np.pi * min(local_phase * 1.8, 1.0))
            
            # Lift feet upward in body frame (positive z offset)
            foot[2] += self.tuck_retraction * tuck_envelope
            
            # Gentle inward retraction in horizontal plane
            foot[0] *= (1.0 - 0.5 * self.tuck_retraction * tuck_envelope)
            foot[1] *= (1.0 - 0.5 * self.tuck_retraction * tuck_envelope)
        
        # Phase 0.75-0.90: Landing preparation - RETURN TO NEUTRAL ONLY (Iteration 2 approach)
        elif phase < 0.90:
            local_phase = (phase - 0.75) / 0.15
            
            # Gradually release tuck back to neutral base position
            tuck_release = 1.0 - local_phase
            remaining_tuck = self.tuck_retraction * np.sin(np.pi * min(tuck_release * 1.8, 1.0))
            foot[2] += remaining_tuck  # ADDITIVE - returns toward zero offset
            
            # Release horizontal retraction
            horizontal_release = self.tuck_retraction * np.sin(np.pi * min(tuck_release * 1.8, 1.0))
            foot[0] *= (1.0 - 0.5 * horizontal_release)
            foot[1] *= (1.0 - 0.5 * horizontal_release)
        
        # Phase 0.90-1.00: Landing and stabilization - gentle settling only
        else:
            local_phase = (phase - 0.90) / 0.10
            # Small settling motion to soften final contact appearance
            settling = 0.012 * np.sin(np.pi * local_phase)
            foot[2] += settling  # ADDITIVE - gentle up/down motion around neutral
        
        return foot