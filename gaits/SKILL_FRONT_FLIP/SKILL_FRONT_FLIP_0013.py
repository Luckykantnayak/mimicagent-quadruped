from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_FRONT_FLIP_MotionGenerator(BaseMotionGenerator):
    """
    Front flip motion with synchronized leg motion and full pitch rotation.
    
    Phase structure:
    - 0.00-0.15: Crouch preparation (all legs retract)
    - 0.15-0.30: Launch and pitch initiation (explosive upward, pitch starts)
    - 0.30-0.70: Airborne rotation (legs tucked, continuous pitch rate)
    - 0.70-0.85: Rotation completion and landing prep (legs extend, pitch rate reduces)
    - 0.85-1.00: Landing and stabilization (contact, absorption, stabilize)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0  # Full flip cycle frequency
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Crouch and tuck parameters
        self.crouch_height = 0.08  # How much to retract legs during crouch
        self.tuck_height = 0.15    # How much to tuck legs toward body during flight
        self.tuck_longitudinal = 0.08  # Longitudinal retraction toward COM during flight
        
        # Launch parameters
        self.launch_vz = 2.5       # Peak upward velocity during launch
        self.launch_vx = 0.1       # Minimal forward drift
        
        # Pitch rotation parameters
        self.peak_pitch_rate = 10.0  # rad/s, tuned for ~360 degree rotation
        
        # Landing absorption
        self.landing_compression = 0.06  # Leg compression during landing
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base linear and angular velocity based on phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        # Phase 0.0-0.15: Crouch preparation
        if phase < 0.15:
            # Minimal motion, slight settling
            vz = -0.2
            pitch_rate = 0.0
        
        # Phase 0.15-0.30: Launch and pitch initiation
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            # Explosive upward velocity
            vz = self.launch_vz * np.sin(np.pi * local_phase)
            # Minimal forward drift
            vx = self.launch_vx
            # Initiate forward pitch rate (ramp up)
            pitch_rate = self.peak_pitch_rate * local_phase
        
        # Phase 0.30-0.70: Airborne rotation
        elif phase < 0.70:
            local_phase = (phase - 0.30) / 0.40
            # Parabolic trajectory (ballistic)
            # Peak at mid-phase, then descend
            vz = self.launch_vz * 0.5 * (1.0 - 2.0 * local_phase)
            vx = self.launch_vx * 0.5
            # Sustained forward pitch rate
            pitch_rate = self.peak_pitch_rate
        
        # Phase 0.70-0.85: Rotation completion and landing prep
        elif phase < 0.85:
            local_phase = (phase - 0.70) / 0.15
            # Descending
            vz = -self.launch_vz * 0.8 * local_phase
            vx = self.launch_vx * 0.2
            # Decrease pitch rate toward zero
            pitch_rate = self.peak_pitch_rate * (1.0 - local_phase)
        
        # Phase 0.85-1.0: Landing and stabilization
        else:
            local_phase = (phase - 0.85) / 0.15
            # Absorb landing, decelerate to zero
            vz = -self.launch_vz * 0.3 * (1.0 - local_phase)
            vx = 0.0
            pitch_rate = 0.0
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        All legs move synchronously.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if front or rear leg for longitudinal adjustments
        is_front = leg_name.startswith('F')
        
        # Phase 0.0-0.15: Crouch preparation
        if phase < 0.15:
            local_phase = phase / 0.15
            # Smooth retraction upward
            retract = self.crouch_height * np.sin(0.5 * np.pi * local_phase)
            foot[2] += retract
            # Slight longitudinal retraction toward COM
            if is_front:
                foot[0] -= 0.03 * local_phase
            else:
                foot[0] += 0.03 * local_phase
        
        # Phase 0.15-0.30: Launch (extension then immediate tuck)
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            if local_phase < 0.5:
                # First half: explosive extension (return toward nominal)
                retract = self.crouch_height * (1.0 - 2.0 * local_phase)
                foot[2] += retract
            else:
                # Second half: begin tucking toward body
                tuck_progress = (local_phase - 0.5) / 0.5
                foot[2] += self.tuck_height * tuck_progress
                if is_front:
                    foot[0] -= self.tuck_longitudinal * tuck_progress
                else:
                    foot[0] += self.tuck_longitudinal * tuck_progress
        
        # Phase 0.30-0.70: Airborne rotation (legs fully tucked)
        elif phase < 0.70:
            # Legs held compact near body COM
            foot[2] += self.tuck_height
            if is_front:
                foot[0] -= self.tuck_longitudinal
            else:
                foot[0] += self.tuck_longitudinal
        
        # Phase 0.70-0.85: Landing preparation (extend legs)
        elif phase < 0.85:
            local_phase = (phase - 0.70) / 0.15
            # Smooth extension back to nominal stance
            tuck_remaining = self.tuck_height * (1.0 - local_phase)
            foot[2] += tuck_remaining
            if is_front:
                foot[0] -= self.tuck_longitudinal * (1.0 - local_phase)
            else:
                foot[0] += self.tuck_longitudinal * (1.0 - local_phase)
        
        # Phase 0.85-1.0: Landing and stabilization
        else:
            local_phase = (phase - 0.85) / 0.15
            # Compliant landing: foot compresses upward then stabilizes
            if local_phase < 0.6:
                # Compression phase
                compression = self.landing_compression * np.sin(np.pi * local_phase / 0.6)
                foot[2] += compression
            else:
                # Stabilization: return to nominal
                compression = self.landing_compression * (1.0 - (local_phase - 0.6) / 0.4)
                foot[2] += compression
        
        return foot