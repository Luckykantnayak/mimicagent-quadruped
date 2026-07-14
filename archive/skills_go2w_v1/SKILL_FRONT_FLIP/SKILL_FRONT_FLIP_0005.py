from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_FRONT_FLIP_MotionGenerator(BaseMotionGenerator):
    """
    Front flip motion: quadruped performs a complete 360-degree forward pitch rotation.
    
    Phase breakdown:
      [0.0, 0.15]: Preparation/crouch - all feet grounded
      [0.15, 0.30]: Launch and liftoff - explosive upward velocity, high pitch rate begins
      [0.30, 0.70]: Airborne rotation - all feet off ground, sustained pitch rotation
      [0.70, 0.85]: Landing preparation - pitch rate decelerates, feet extend
      [0.85, 1.0]: Stabilization - all feet grounded, return to neutral stance
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Complete flip takes 2 seconds at standard playback
        
        # Store nominal foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Flip parameters
        self.crouch_depth = 0.05  # Body lowers 5cm during preparation
        self.launch_velocity_z = 2.5  # Upward launch velocity (m/s)
        self.peak_pitch_rate = 12.0  # Peak forward pitch rate (rad/s) ~2 rev/s
        self.tuck_height = 0.15  # How much feet retract toward body during flip
        self.tuck_forward = 0.12  # Front legs tuck rearward, rear legs tuck forward
        self.backward_drift_compensation = -0.3  # Small backward velocity to counter pitch-induced drift
        
        # Track cumulative pitch for debugging/validation
        self.cumulative_pitch = 0.0

    def reset(self, root_pos, root_quat):
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0
        self.cumulative_pitch = 0.0

    def update_base_motion(self, phase, dt):
        """
        Prescribe base velocities to execute front flip kinematically.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 1: Preparation and crouch [0.0, 0.15]
        if phase < 0.15:
            local_phase = phase / 0.15
            # Crouch: downward then back to zero
            if local_phase < 0.5:
                vz = -0.4  # Downward velocity
            else:
                vz = 0.2  # Return upward slightly
            # Start small forward pitch rate
            pitch_rate = 1.0
        
        # Phase 2: Launch and liftoff [0.15, 0.30]
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            # Explosive upward velocity
            vz = self.launch_velocity_z * (1.0 - 0.3 * local_phase)  # Slightly decaying
            # High pitch rate begins
            pitch_rate = self.peak_pitch_rate * (0.5 + 0.5 * local_phase)  # Ramp up
            # Slight backward drift compensation
            vx = self.backward_drift_compensation * 0.5
        
        # Phase 3: Airborne rotation [0.30, 0.70]
        elif phase < 0.70:
            local_phase = (phase - 0.30) / 0.40
            # Parabolic z-velocity: starts positive, becomes negative
            apex_phase = 0.4  # Apex occurs 40% through airborne phase
            if local_phase < apex_phase:
                # Ascending
                vz = self.launch_velocity_z * 0.7 * (1.0 - local_phase / apex_phase)
            else:
                # Descending
                desc_progress = (local_phase - apex_phase) / (1.0 - apex_phase)
                vz = -self.launch_velocity_z * 0.8 * desc_progress
            
            # Sustained high pitch rate
            pitch_rate = self.peak_pitch_rate
            
            # Backward drift compensation
            vx = self.backward_drift_compensation
        
        # Phase 4: Landing preparation [0.70, 0.85]
        elif phase < 0.85:
            local_phase = (phase - 0.70) / 0.15
            # Descending, decelerating
            vz = -self.launch_velocity_z * 0.8 * (1.0 - 0.6 * local_phase)
            # Pitch rate decelerates rapidly to zero
            pitch_rate = self.peak_pitch_rate * (1.0 - local_phase)**2
            # Reduce backward drift
            vx = self.backward_drift_compensation * (1.0 - local_phase)
        
        # Phase 5: Stabilization [0.85, 1.0]
        else:
            local_phase = (phase - 0.85) / 0.15
            # Arrest downward velocity
            vz = -0.5 * (1.0 - local_phase)
            # Drive pitch rate to zero
            pitch_rate = 0.0
            vx = 0.0
        
        # Set velocity commands
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
        
        # Track cumulative pitch for validation
        self.cumulative_pitch += pitch_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot trajectory in body frame throughout flip phases.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Phase 1: Preparation and crouch [0.0, 0.15]
        if phase < 0.15:
            # Feet remain on ground at nominal stance position
            # As body crouches, foot z in body frame increases slightly
            local_phase = phase / 0.15
            if local_phase < 0.5:
                foot[2] += self.crouch_depth * local_phase / 0.5
            else:
                foot[2] += self.crouch_depth * (1.0 - (local_phase - 0.5) / 0.5)
        
        # Phase 2: Launch and liftoff [0.15, 0.30]
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            # Feet retract toward body center (tucking begins)
            tuck_progress = local_phase
            
            # Retract upward (reduce z)
            foot[2] -= self.tuck_height * tuck_progress
            
            # Front legs tuck rearward (x increases toward 0)
            # Rear legs tuck forward (x decreases toward 0)
            if is_front:
                foot[0] += self.tuck_forward * tuck_progress
            else:
                foot[0] -= self.tuck_forward * tuck_progress
        
        # Phase 3: Airborne rotation [0.30, 0.70]
        elif phase < 0.70:
            # Feet remain fully tucked close to body centerline
            foot[2] -= self.tuck_height
            if is_front:
                foot[0] += self.tuck_forward
            else:
                foot[0] -= self.tuck_forward
        
        # Phase 4: Landing preparation [0.70, 0.85]
        elif phase < 0.85:
            local_phase = (phase - 0.70) / 0.15
            # Feet extend back toward nominal stance positions
            extension_progress = local_phase
            
            # Extend downward (increase z back to nominal)
            foot[2] -= self.tuck_height * (1.0 - extension_progress)
            
            # Front legs extend forward, rear legs extend rearward
            if is_front:
                foot[0] += self.tuck_forward * (1.0 - extension_progress)
            else:
                foot[0] -= self.tuck_forward * (1.0 - extension_progress)
        
        # Phase 5: Stabilization [0.85, 1.0]
        else:
            # Feet at nominal stance position, grounded
            foot = base_pos.copy()
        
        return foot