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
        
        # Motion parameters tuned to stay within constraints
        self.crouch_depth = 0.07
        self.launch_height = 0.08
        self.tuck_amount = 0.06
        self.landing_extension = 0.09  # Increased to prevent ground penetration
        
        # Base velocities tuned for height constraint and smooth landing
        self.launch_vz = 1.0  # Reduced to keep peak height under 0.68m
        self.launch_pitch_rate = 9.0
        self.aerial_pitch_rate = 6.5  # Reduced to prevent overshoot
        self.gravity_vz = -0.8  # Reduced for gentler landing
        
    def update_base_motion(self, phase, dt):
        """
        Update base motion through all phases of the front flip.
        """
        
        # Phase 1: Preparation crouch [0.0, 0.15]
        if phase < 0.15:
            progress = phase / 0.15
            smooth_progress = self._smooth_step(progress)
            vz = -0.4 * np.sin(np.pi * smooth_progress)
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, 0.0, 0.0])
        
        # Phase 2: Launch and takeoff [0.15, 0.35]
        elif phase < 0.35:
            progress = (phase - 0.15) / 0.20
            smooth_progress = self._smooth_step(progress)
            # Gradual velocity buildup and decay
            vz = self.launch_vz * np.sin(np.pi * smooth_progress * 0.65)
            pitch_rate = self.launch_pitch_rate * smooth_progress
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 3: Aerial rotation [0.35, 0.70]
        elif phase < 0.70:
            progress = (phase - 0.35) / 0.35
            # Aggressive early transition to descent
            vz_blend = progress ** 1.5  # Steeper curve for faster descent
            vz = self.launch_vz * 0.15 * (1.0 - vz_blend) + self.gravity_vz * vz_blend
            pitch_rate = self.aerial_pitch_rate
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 4: Landing preparation [0.70, 0.80]
        elif phase < 0.80:
            progress = (phase - 0.70) / 0.10
            smooth_progress = self._smooth_step(progress)
            # Gentler descent for landing
            vz = self.gravity_vz * 0.6 * (1.0 - 0.5 * smooth_progress)
            # Decelerate pitch rate to zero by phase 0.78
            pitch_decay = max(0.0, 1.0 - progress * 1.5)
            pitch_rate = self.aerial_pitch_rate * pitch_decay
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 5: Impact and stabilization [0.80, 1.0]
        else:
            progress = (phase - 0.80) / 0.20
            smooth_progress = self._smooth_step(progress)
            # Rapid deceleration
            decay = 1.0 - smooth_progress
            vz = self.gravity_vz * 0.3 * decay * decay  # Quadratic decay for faster stop
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
        Conservative trajectories to avoid joint limits and ground penetration.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        is_rear = leg_name.startswith('R')
        
        # Phase 1: Preparation crouch [0.0, 0.15]
        if phase < 0.15:
            progress = phase / 0.15
            smooth_progress = self._smooth_step(progress)
            retraction = self.crouch_depth * smooth_progress
            foot[2] += retraction
        
        # Phase 2: Launch and takeoff [0.15, 0.35]
        elif phase < 0.35:
            progress = (phase - 0.15) / 0.20
            smooth_progress = self._smooth_step(progress)
            
            if is_rear:
                # Smoother, smaller extension
                extension_curve = np.sin(np.pi * smooth_progress)
                extension = -self.launch_height * extension_curve
                foot[2] += self.crouch_depth + extension
                foot[0] += 0.01 * extension_curve
            else:
                # Front legs: gradual retraction
                retract_z = self.crouch_depth + self.tuck_amount * smooth_progress * 0.8
                retract_x = -0.03 * smooth_progress
                foot[2] += retract_z
                foot[0] += retract_x
        
        # Phase 3: Aerial rotation [0.35, 0.70]
        elif phase < 0.70:
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
            smooth_progress = self._smooth_step(progress)
            
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
            
            # Delayed compression to allow base to stabilize first
            if progress < 0.3:
                # Maintain extended position
                foot[2] += -self.landing_extension
            elif progress < 0.65:
                # Gradual compression
                local_progress = (progress - 0.3) / 0.35
                smooth_local = self._smooth_step(local_progress)
                compression = self.crouch_depth * 0.4 * smooth_local
                foot[2] += -self.landing_extension + compression
            else:
                # Recovery to neutral
                local_progress = (progress - 0.65) / 0.35
                smooth_local = self._smooth_step(local_progress)
                max_compression = self.crouch_depth * 0.4
                compression = max_compression * (1.0 - smooth_local)
                foot[2] += -self.landing_extension + compression
        
        return foot
    
    def _smooth_step(self, t):
        """Smoothstep function for C1 continuity: 3t^2 - 2t^3"""
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)