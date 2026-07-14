from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SPLIT_APEX_REENTRY_MotionGenerator(BaseMotionGenerator):
    """
    Complex jump skill with staged reentry:
    - Explosive takeoff with forward bias
    - Full aerial phase with leg retraction
    - Distal-only contact for horizontal momentum dissipation
    - Delayed full limb touchdown and stabilization
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower cycle for complex aerial maneuver
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Takeoff parameters
        self.takeoff_extension = 0.07
        self.takeoff_vx_peak = 2.5
        self.takeoff_vz_peak = 1.3
        
        # Aerial retraction parameters
        self.retraction_height = 0.13
        self.retraction_forward = 0.10
        
        # Distal contact parameters - increased to maintain clearance during descent
        self.distal_height_offset = 0.22  # Increased from 0.06 to prevent ground penetration
        self.distal_slide_distance = 0.20
        
        # Landing parameters
        self.landing_settle_height = 0.04
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion using phase-dependent velocity profiles.
        Implements ballistic trajectory with staged reentry.
        Reduced descent rates to maintain safe base height.
        """
        
        # Phase 0.0-0.2: Explosive takeoff
        if phase < 0.2:
            progress = phase / 0.2
            vx = self.takeoff_vx_peak * np.sin(np.pi * progress / 2.0)
            vz = self.takeoff_vz_peak * np.sin(np.pi * progress / 2.0)
            pitch_rate = 0.8 * np.sin(np.pi * progress)
            
            self.vel_world = np.array([vx, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 0.2-0.45: Aerial ascent to apex (restored to skill spec)
        elif phase < 0.45:
            progress = (phase - 0.2) / 0.25
            vx = self.takeoff_vx_peak * (1.0 - 0.2 * progress)
            # Reduced vertical decay rate
            vz = self.takeoff_vz_peak * (1.0 - progress) - 0.8 * progress
            pitch_rate = -0.6 * np.sin(np.pi * progress)
            
            self.vel_world = np.array([vx, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 0.45-0.6: Distal initial contact (per skill spec, controlled descent)
        elif phase < 0.6:
            progress = (phase - 0.45) / 0.15
            vx = self.takeoff_vx_peak * 0.8 * (1.0 - 0.3 * progress)
            # Reduced descent rate to prevent base envelope violation
            vz = -0.6 - 0.2 * progress
            pitch_rate = 0.4 * np.sin(np.pi * progress / 2.0)
            
            self.vel_world = np.array([vx, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 0.6-0.8: Momentum dissipation (per skill spec)
        elif phase < 0.8:
            progress = (phase - 0.6) / 0.2
            vx = self.takeoff_vx_peak * 0.56 * (1.0 - progress) * np.exp(-2.0 * progress)
            # Gentler descent rate approaching zero
            vz = -0.8 + 0.6 * progress
            pitch_rate = -0.5 * (1.0 - progress)
            
            self.vel_world = np.array([vx, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 0.8-1.0: Delayed limb touchdown and stabilization (per skill spec)
        else:
            progress = (phase - 0.8) / 0.2
            vx = self.takeoff_vx_peak * 0.06 * (1.0 - progress)**2
            # Smooth decay to zero
            vz = -0.2 * (1.0 - progress)**2
            pitch_rate = 0.0
            
            self.vel_world = np.array([vx, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
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
        Compute foot trajectory for synchronized four-leg jump with staged reentry.
        All legs move synchronously through takeoff, aerial retraction, distal contact, and full touchdown.
        Enhanced foot elevation during distal contact to maintain ground clearance.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if front or rear leg for subtle forward/backward bias
        is_front = leg_name.startswith('F')
        front_bias = 0.015 if is_front else -0.015
        
        # Phase 0.0-0.2: Explosive takeoff (stance pushing)
        if phase < 0.2:
            progress = phase / 0.2
            foot[2] -= self.takeoff_extension * np.sin(np.pi * progress / 2.0)
            foot[0] -= 0.025 * progress
        
        # Phase 0.2-0.45: Aerial ascent with retraction (per skill spec)
        elif phase < 0.45:
            progress = (phase - 0.2) / 0.25
            # Smooth cubic blend from takeoff to retraction
            blend = progress**2 * (3.0 - 2.0 * progress)
            
            # Decay takeoff extension
            extension_decay = self.takeoff_extension * np.cos(np.pi * blend / 2.0)**2
            foot[2] -= extension_decay
            
            # Retract toward body COM
            foot[2] += self.retraction_height * blend
            foot[0] += (self.retraction_forward + front_bias) * blend
        
        # Phase 0.45-0.8: Distal contact and momentum dissipation (combined per skill spec)
        elif phase < 0.8:
            progress = (phase - 0.45) / 0.35
            
            # Maintain retracted height from aerial phase
            foot[2] += self.retraction_height
            
            # Additional distal elevation to prevent ground penetration during base descent
            # Smoothly transition from max elevation to reduced elevation
            distal_extra = self.distal_height_offset * (1.0 - 0.5 * progress)
            foot[2] += distal_extra
            
            # Forward position at start of distal contact
            initial_forward = self.retraction_forward + front_bias
            
            # Rearward drift as momentum dissipates
            slide = self.distal_slide_distance * progress
            foot[0] += initial_forward - slide
            
            # Subtle vertical oscillation for tangential sliding dynamics
            foot[2] -= 0.015 * np.sin(2.0 * np.pi * progress)
        
        # Phase 0.8-1.0: Delayed full limb touchdown and stabilization (per skill spec)
        else:
            progress = (phase - 0.8) / 0.2
            
            # Start from retained retraction height plus remaining distal offset
            start_height = self.retraction_height + self.distal_height_offset * 0.5
            
            # Smooth descent to ground with cubic easing
            height_blend = progress**2 * (3.0 - 2.0 * progress)
            foot[2] += start_height * (1.0 - height_blend)
            
            # Return to neutral x position
            end_x_offset = self.retraction_forward + front_bias - self.distal_slide_distance
            foot[0] += end_x_offset * (1.0 - height_blend)
            
            # Subtle settling oscillation
            foot[2] -= self.landing_settle_height * (1.0 - progress) * np.exp(-3.0 * progress)
        
        return foot