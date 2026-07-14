from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_FRONT_FLIP_MotionGenerator(BaseMotionGenerator):
    """
    Front flip motion: quadruped performs a complete 360-degree forward pitch rotation.
    
    Phase breakdown:
      [0.0, 0.15]: Preparation/crouch - all feet grounded
      [0.15, 0.30]: Launch and liftoff - controlled upward velocity, pitch rate begins
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
        
        # Flip parameters - significantly reduced from original
        self.crouch_depth = 0.02  # Minimal crouch - 2cm body lowering
        self.launch_velocity_z = 1.1  # Reduced from 2.5 to 1.1 m/s
        self.peak_pitch_rate = 11.5  # Slightly reduced for control
        self.tuck_height = 0.06  # Reduced from 0.15m to 0.06m
        self.tuck_forward = 0.04  # Reduced from 0.12m to 0.04m
        self.backward_drift_compensation = -0.15  # Reduced compensation
        
        # Track cumulative pitch
        self.cumulative_pitch = 0.0

    def reset(self, root_pos, root_quat):
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0
        self.cumulative_pitch = 0.0

    def update_base_motion(self, phase, dt):
        """
        Prescribe base velocities to execute front flip kinematically.
        Reduced launch velocity and more controlled trajectory to stay within height envelope.
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
            # Minimal crouch with smooth profile
            if local_phase < 0.6:
                vz = -0.25 * np.sin(local_phase * np.pi / 0.6)
            else:
                vz = 0.15 * np.sin((local_phase - 0.6) * np.pi / 0.4)
            # Start small forward pitch rate
            pitch_rate = 1.5 * local_phase
        
        # Phase 2: Launch and liftoff [0.15, 0.30]
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            # Controlled upward velocity - smooth ramp up then decay
            vz = self.launch_velocity_z * np.sin(local_phase * np.pi)
            # Pitch rate ramps up smoothly
            pitch_rate = self.peak_pitch_rate * (0.3 + 0.7 * local_phase)
            # Slight backward drift compensation
            vx = self.backward_drift_compensation * 0.4 * local_phase
        
        # Phase 3: Airborne rotation [0.30, 0.70]
        elif phase < 0.70:
            local_phase = (phase - 0.30) / 0.40
            # Parabolic z-velocity: peak earlier and descend faster
            apex_phase = 0.25  # Apex occurs 25% through airborne phase
            if local_phase < apex_phase:
                # Ascending - decaying upward velocity
                vz = self.launch_velocity_z * 0.5 * (1.0 - local_phase / apex_phase)
            else:
                # Descending - stronger downward velocity
                desc_progress = (local_phase - apex_phase) / (1.0 - apex_phase)
                vz = -self.launch_velocity_z * 1.3 * desc_progress
            
            # Sustained high pitch rate with smooth envelope
            pitch_rate = self.peak_pitch_rate * (1.0 - 0.1 * local_phase)
            
            # Backward drift compensation
            vx = self.backward_drift_compensation * (1.0 - 0.3 * local_phase)
        
        # Phase 4: Landing preparation [0.70, 0.85]
        elif phase < 0.85:
            local_phase = (phase - 0.70) / 0.15
            # Descending with smooth deceleration
            vz = -self.launch_velocity_z * 1.0 * (1.0 - 0.7 * local_phase)
            # Pitch rate decelerates smoothly to zero
            pitch_rate = self.peak_pitch_rate * 0.9 * (1.0 - local_phase) ** 2.5
            # Reduce backward drift
            vx = self.backward_drift_compensation * 0.3 * (1.0 - local_phase)
        
        # Phase 5: Stabilization [0.85, 1.0]
        else:
            local_phase = (phase - 0.85) / 0.15
            # Arrest downward velocity smoothly
            vz = -0.3 * (1.0 - local_phase) ** 2
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
        
        # Track cumulative pitch
        self.cumulative_pitch += pitch_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot trajectory in body frame throughout flip phases.
        Reduced tuck amplitude and improved ground contact handling.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Phase 1: Preparation and crouch [0.0, 0.15]
        if phase < 0.15:
            local_phase = phase / 0.15
            # Feet stay grounded - no upward adjustment during crouch
            # Body descends relative to fixed feet, so feet appear to move down in body frame
            crouch_factor = np.sin(local_phase * np.pi)
            foot[2] -= self.crouch_depth * crouch_factor * 0.5
        
        # Phase 2: Launch and liftoff [0.15, 0.30]
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            # Feet retract toward body center with smooth profile
            tuck_progress = 0.5 * (1.0 - np.cos(local_phase * np.pi))
            
            # Retract upward (reduce z magnitude)
            foot[2] += self.tuck_height * tuck_progress
            
            # Front legs tuck slightly rearward, rear legs tuck slightly forward
            if is_front:
                foot[0] += self.tuck_forward * tuck_progress
            else:
                foot[0] -= self.tuck_forward * tuck_progress
        
        # Phase 3: Airborne rotation [0.30, 0.70]
        elif phase < 0.70:
            # Feet remain in tucked position
            foot[2] += self.tuck_height
            if is_front:
                foot[0] += self.tuck_forward
            else:
                foot[0] -= self.tuck_forward
        
        # Phase 4: Landing preparation [0.70, 0.85]
        elif phase < 0.85:
            local_phase = (phase - 0.70) / 0.15
            # Feet extend back toward nominal stance positions with smooth profile
            extension_progress = 0.5 * (1.0 - np.cos(local_phase * np.pi))
            
            # Extend downward (return z to nominal)
            foot[2] += self.tuck_height * (1.0 - extension_progress)
            
            # Front legs extend forward, rear legs extend rearward
            if is_front:
                foot[0] += self.tuck_forward * (1.0 - extension_progress)
            else:
                foot[0] -= self.tuck_forward * (1.0 - extension_progress)
        
        # Phase 5: Stabilization [0.85, 1.0]
        else:
            # Feet at nominal stance position
            foot = base_pos.copy()
        
        return foot

    def get_foot_contact_state(self, leg_name, phase):
        """
        Return expected contact state for each leg throughout the motion.
        """
        # Stance during prep and stabilization
        if phase < 0.15 or phase >= 0.85:
            return True
        # Swing/airborne during launch, rotation, and landing prep
        else:
            return False

    def __call__(self, phase, dt):
        self.t += dt
        self.update_base_motion(phase, dt)
        
        foot_positions = {}
        for leg_name in self.leg_names:
            foot_positions[leg_name] = self.compute_foot_position_body_frame(leg_name, phase)
        
        return {
            "root_pos": self.root_pos.copy(),
            "root_quat": self.root_quat.copy(),
            "foot_positions_body": foot_positions,
            "vel_world": self.vel_world.copy(),
            "omega_world": self.omega_world.copy(),
        }