from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_FRONT_FLIP_MotionGenerator(BaseMotionGenerator):
    """
    Front flip motion: The robot performs a complete forward somersault.
    
    Phase breakdown:
      [0.0, 0.15]: Preparation crouch - all legs compress, base lowers
      [0.15, 0.35]: Launch and takeoff - rear legs extend smoothly, front legs retract, base lifts off
      [0.35, 0.70]: Aerial rotation - all feet off ground, body rotates forward ~360 degrees
      [0.70, 0.80]: Landing preparation - legs extend anticipatorily, still airborne
      [0.80, 1.0]: Impact and stabilization - all legs contact ground, absorb impact, return to stance
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5
        
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Reduced motion parameters to stay within constraints
        self.crouch_depth = 0.06
        self.launch_height = 0.08  # Reduced from 0.20
        self.tuck_amount = 0.06  # Reduced from 0.12
        self.landing_extension = 0.04
        
        # Reduced base velocities to keep height under 0.68m
        self.launch_vz = 1.2  # Reduced from 2.5
        self.launch_pitch_rate = 9.0  # Increased to maintain rotation
        self.aerial_pitch_rate = 7.0  # Increased to maintain rotation
        self.gravity_vz = -1.2
        
    def update_base_motion(self, phase, dt):
        """
        Update base motion through all phases of the front flip.
        """
        
        # Phase 1: Preparation crouch [0.0, 0.15]
        if phase < 0.15:
            progress = phase / 0.15
            smooth_progress = smooth_step(progress)
            vz = -0.4 * np.sin(np.pi * smooth_progress)
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, 0.0, 0.0])
        
        # Phase 2: Launch and takeoff [0.15, 0.35] - extended duration
        elif phase < 0.35:
            progress = (phase - 0.15) / 0.20
            smooth_progress = smooth_step(progress)
            # Gradual velocity buildup and decay
            vz = self.launch_vz * np.sin(np.pi * smooth_progress * 0.6)
            pitch_rate = self.launch_pitch_rate * smooth_progress
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 3: Aerial rotation [0.35, 0.70]
        elif phase < 0.70:
            progress = (phase - 0.35) / 0.35
            # Early transition to descent
            vz_blend = progress ** 2  # Quadratic for faster descent transition
            vz = self.launch_vz * 0.3 * (1.0 - vz_blend) + self.gravity_vz * vz_blend
            pitch_rate = self.aerial_pitch_rate
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 4: Landing preparation [0.70, 0.80]
        elif phase < 0.80:
            progress = (phase - 0.70) / 0.10
            vz = self.gravity_vz * (1.0 - 0.3 * progress)
            pitch_rate = self.aerial_pitch_rate * (1.0 - smooth_step(progress))
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 5: Impact and stabilization [0.80, 1.0]
        else:
            progress = (phase - 0.80) / 0.20
            smooth_progress = smooth_step(progress)
            decay = 1.0 - smooth_progress
            vz = self.gravity_vz * 0.2 * decay
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for each leg through all phases.
        Conservative trajectories to avoid joint limits.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        is_rear = leg_name.startswith('R')
        
        # Phase 1: Preparation crouch [0.0, 0.15]
        if phase < 0.15:
            progress = phase / 0.15
            smooth_progress = smooth_step(progress)
            retraction = self.crouch_depth * smooth_progress
            foot[2] += retraction
        
        # Phase 2: Launch and takeoff [0.15, 0.35] - extended and smoothed
        elif phase < 0.35:
            progress = (phase - 0.15) / 0.20
            smooth_progress = smooth_step(progress)
            
            if is_rear:
                # Smoother, smaller extension
                extension_curve = np.sin(np.pi * smooth_progress)
                extension = -self.launch_height * extension_curve
                foot[2] += self.crouch_depth + extension
                # Minimal horizontal shift
                foot[0] += 0.01 * extension_curve
            else:
                # Front legs: gradual retraction
                retract_z = self.crouch_depth + self.tuck_amount * smooth_progress * 0.8
                retract_x = -0.03 * smooth_progress
                foot[2] += retract_z
                foot[0] += retract_x
        
        # Phase 3: Aerial rotation [0.35, 0.70]
        elif phase < 0.70:
            progress = (phase - 0.35) / 0.35
            smooth_progress = smooth_step(progress)
            
            # Conservative tuck - legs stay closer to neutral
            if is_front:
                tuck_z = self.crouch_depth + self.tuck_amount * 0.6
                tuck_x = -0.03
                foot[2] += tuck_z
                foot[0] += tuck_x
            else:
                tuck_z = self.tuck_amount * 0.5
                tuck_x = 0.02
                foot[2] += tuck_z
                foot[0] += tuck_x
        
        # Phase 4: Landing preparation [0.70, 0.80]
        elif phase < 0.80:
            progress = (phase - 0.70) / 0.10
            smooth_progress = smooth_step(progress)
            
            if is_front:
                start_z = self.crouch_depth + self.tuck_amount * 0.6
                start_x = -0.03
                target_z = -self.landing_extension
                foot[2] += start_z + (target_z - start_z) * smooth_progress
                foot[0] += start_x * (1.0 - smooth_progress)
            else:
                start_z = self.tuck_amount * 0.5
                start_x = 0.02
                target_z = -self.landing_extension
                foot[2] += start_z + (target_z - start_z) * smooth_progress
                foot[0] += start_x * (1.0 - smooth_progress)
        
        # Phase 5: Impact and stabilization [0.80, 1.0]
        else:
            progress = (phase - 0.80) / 0.20
            
            # Smooth compression and recovery
            if progress < 0.4:
                # Compression
                local_progress = progress / 0.4
                smooth_local = smooth_step(local_progress)
                compression = self.crouch_depth * 0.5 * smooth_local
                foot[2] += -self.landing_extension + compression
            else:
                # Recovery to neutral
                local_progress = (progress - 0.4) / 0.6
                smooth_local = smooth_step(local_progress)
                max_compression = self.crouch_depth * 0.5
                compression = max_compression * (1.0 - smooth_local)
                foot[2] += -self.landing_extension + compression
        
        return foot


def smooth_step(t):
    """Smoothstep function for C1 continuity: 3t^2 - 2t^3"""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)