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
        self.freq = 1.0
        
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        self.crouch_height = 0.08
        self.tuck_height = 0.15
        self.tuck_longitudinal = 0.08
        
        self.launch_vz = 1.85
        self.launch_vx = 0.1
        
        self.peak_pitch_rate = 10.0
        
        self.landing_compression = 0.06
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base linear and angular velocity based on phase.
        Vertical velocity tuned to respect 0.68m maximum height constraint.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        if phase < 0.15:
            vz = -0.2
            pitch_rate = 0.0
        
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            vz = self.launch_vz * np.sin(np.pi * local_phase)
            vx = self.launch_vx
            pitch_rate = self.peak_pitch_rate * local_phase
        
        elif phase < 0.70:
            local_phase = (phase - 0.30) / 0.40
            apex_phase = 0.35
            if local_phase < apex_phase:
                ascent_progress = local_phase / apex_phase
                vz = self.launch_vz * 0.4 * (1.0 - ascent_progress)
            else:
                descent_progress = (local_phase - apex_phase) / (1.0 - apex_phase)
                vz = -self.launch_vz * 0.6 * descent_progress
            
            vx = self.launch_vx * 0.5
            pitch_rate = self.peak_pitch_rate
        
        elif phase < 0.85:
            local_phase = (phase - 0.70) / 0.15
            vz = -self.launch_vz * 0.8 * local_phase
            vx = self.launch_vx * 0.2
            pitch_rate = self.peak_pitch_rate * (1.0 - local_phase)
        
        else:
            local_phase = (phase - 0.85) / 0.15
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
        
        is_front = leg_name.startswith('F')
        
        if phase < 0.15:
            local_phase = phase / 0.15
            retract = self.crouch_height * np.sin(0.5 * np.pi * local_phase)
            foot[2] += retract
            if is_front:
                foot[0] -= 0.03 * local_phase
            else:
                foot[0] += 0.03 * local_phase
        
        elif phase < 0.30:
            local_phase = (phase - 0.15) / 0.15
            if local_phase < 0.5:
                retract = self.crouch_height * (1.0 - 2.0 * local_phase)
                foot[2] += retract
            else:
                tuck_progress = (local_phase - 0.5) / 0.5
                foot[2] += self.tuck_height * tuck_progress
                if is_front:
                    foot[0] -= self.tuck_longitudinal * tuck_progress
                else:
                    foot[0] += self.tuck_longitudinal * tuck_progress
        
        elif phase < 0.70:
            foot[2] += self.tuck_height
            if is_front:
                foot[0] -= self.tuck_longitudinal
            else:
                foot[0] += self.tuck_longitudinal
        
        elif phase < 0.85:
            local_phase = (phase - 0.70) / 0.15
            tuck_remaining = self.tuck_height * (1.0 - local_phase)
            foot[2] += tuck_remaining
            if is_front:
                foot[0] -= self.tuck_longitudinal * (1.0 - local_phase)
            else:
                foot[0] += self.tuck_longitudinal * (1.0 - local_phase)
        
        else:
            local_phase = (phase - 0.85) / 0.15
            if local_phase < 0.6:
                compression = self.landing_compression * np.sin(np.pi * local_phase / 0.6)
                foot[2] += compression
            else:
                compression = self.landing_compression * (1.0 - (local_phase - 0.6) / 0.4)
                foot[2] += compression
        
        return foot