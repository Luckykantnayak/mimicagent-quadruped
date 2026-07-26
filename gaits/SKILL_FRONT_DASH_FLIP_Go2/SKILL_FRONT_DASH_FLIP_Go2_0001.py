from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_FRONT_DASH_FLIP_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Front dash flip: explosive launch, aerial tuck and forward flip, landing.
    
    Phase structure:
      [0.0, 0.3]: crouch_and_launch - crouch down, explosive push-off
      [0.3, 0.7]: aerial_tuck_and_flip - all legs tucked, body rotating forward
      [0.7, 1.0]: extend_and_land - legs extend, landing and absorption
    
    All four legs synchronized throughout entire motion.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.crouch_depth = 0.15          # how much body descends during crouch (z direction)
        self.tuck_height = 0.25           # how high feet retract during aerial tuck
        self.tuck_inward = 0.08           # how much feet move toward body center during tuck
        self.landing_forward_reach = 0.12 # how far forward feet reach during landing extension
        
        # Velocity parameters for base motion
        self.vx_launch = 2.5              # forward velocity during launch
        self.vz_launch = 3.0              # upward velocity during launch
        self.pitch_rate_aerial = 6.28     # pitch angular velocity during flip (approx 360 deg/cycle)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity state
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion through three phases: crouch/launch, aerial flip, landing.
        """
        
        if phase < 0.15:
            # Early crouch: body descends, slight backward lean
            progress = phase / 0.15
            vx = -0.3 * (1.0 - progress)
            vz = -1.5 * np.sin(np.pi * progress)
            pitch_rate = 0.0
            
        elif phase < 0.3:
            # Explosive launch: rapid upward and forward acceleration, pitch rotation starts
            progress = (phase - 0.15) / 0.15
            smooth_ramp = np.sin(np.pi * 0.5 * progress)
            vx = self.vx_launch * smooth_ramp
            vz = self.vz_launch * smooth_ramp
            pitch_rate = self.pitch_rate_aerial * smooth_ramp
            
        elif phase < 0.7:
            # Aerial phase: maintain forward velocity, ballistic arc, sustained pitch rotation
            progress = (phase - 0.3) / 0.4
            vx = self.vx_launch * (1.0 - 0.3 * progress)  # slight decay
            # Parabolic z trajectory: up then down
            vz = self.vz_launch * (1.0 - 2.0 * progress) - 2.0 * progress
            pitch_rate = self.pitch_rate_aerial
            
        else:
            # Landing phase: decelerate vertical, maintain forward, slow pitch rotation
            progress = (phase - 0.7) / 0.3
            vx = self.vx_launch * 0.7 * (1.0 - 0.3 * progress)
            vz = -2.0 + 2.0 * progress  # decelerate downward velocity to zero
            pitch_rate = self.pitch_rate_aerial * (1.0 - progress)
        
        self.vel_world = np.array([vx, 0.0, vz])
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
        Compute foot position in body frame for given phase.
        All legs move synchronously through crouch, tuck, and landing.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a front or rear leg (for forward/backward offsets)
        is_front = leg_name.startswith('F')
        forward_sign = 1.0 if is_front else -1.0
        
        if phase < 0.15:
            # Crouch phase: feet move down and inward as body descends
            progress = phase / 0.15
            smooth = np.sin(np.pi * 0.5 * progress)
            foot[2] += self.crouch_depth * smooth  # down (positive z in body frame during crouch)
            foot[0] -= forward_sign * 0.02 * smooth  # slight inward toward body center
            
        elif phase < 0.3:
            # Push-off phase: feet extend rapidly downward then leave ground
            progress = (phase - 0.15) / 0.15
            smooth = np.sin(np.pi * 0.5 * progress)
            foot[2] += self.crouch_depth * (1.0 - smooth)  # return from crouch
            # At end of this phase, feet break contact
            
        elif phase < 0.7:
            # Aerial tuck phase: feet retract tightly toward body center
            progress = (phase - 0.3) / 0.4
            # Smooth entry and hold during middle of aerial phase
            if progress < 0.3:
                tuck_progress = progress / 0.3
            else:
                tuck_progress = 1.0
            
            smooth_tuck = np.sin(np.pi * 0.5 * tuck_progress)
            foot[2] -= self.tuck_height * smooth_tuck  # retract upward
            foot[0] -= forward_sign * self.tuck_inward * smooth_tuck  # pull toward center
            foot[1] *= (1.0 - 0.3 * smooth_tuck)  # pull lateral feet inward
            
        else:
            # Landing extension phase: feet extend downward and forward
            progress = (phase - 0.7) / 0.3
            smooth = np.sin(np.pi * 0.5 * progress)
            
            # Start from tucked position, extend to landing position
            tuck_release = 1.0 - smooth
            foot[2] -= self.tuck_height * tuck_release  # release from tuck
            foot[0] -= forward_sign * self.tuck_inward * tuck_release
            foot[1] *= (1.0 - 0.3 * tuck_release)
            
            # Reach forward and down for landing
            foot[0] += forward_sign * self.landing_forward_reach * smooth
            foot[2] += 0.05 * smooth  # slight downward reach
        
        return foot