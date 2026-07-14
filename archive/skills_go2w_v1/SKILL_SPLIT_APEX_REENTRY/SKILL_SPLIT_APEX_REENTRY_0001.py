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
        self.takeoff_extension = 0.15  # Maximum leg extension during push
        self.takeoff_vx_peak = 2.5  # Forward velocity at takeoff
        self.takeoff_vz_peak = 3.0  # Upward velocity at takeoff
        
        # Aerial retraction parameters
        self.retraction_height = 0.25  # Leg retraction toward body COM
        self.retraction_forward = 0.12  # Forward positioning of distal elements
        
        # Distal contact parameters
        self.distal_height_offset = 0.18  # Height of main limb during distal contact
        self.distal_slide_distance = 0.20  # Rearward drift during momentum dissipation
        
        # Landing parameters
        self.landing_settle_height = 0.05  # Final settling displacement
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (will be set per phase)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion using phase-dependent velocity profiles.
        Implements ballistic trajectory with staged reentry.
        """
        
        # Phase 0.0-0.2: Explosive takeoff
        if phase < 0.2:
            progress = phase / 0.2
            # Rapid acceleration to peak velocities
            vx = self.takeoff_vx_peak * np.sin(np.pi * progress / 2.0)
            vz = self.takeoff_vz_peak * np.sin(np.pi * progress / 2.0)
            # Small positive pitch for forward bias
            pitch_rate = 0.8 * np.sin(np.pi * progress)
            
            self.vel_world = np.array([vx, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 0.2-0.45: Aerial ascent to apex
        elif phase < 0.45:
            progress = (phase - 0.2) / 0.25
            # Ballistic decay: forward velocity persists, vertical decreases
            vx = self.takeoff_vx_peak * (1.0 - 0.3 * progress)
            # Parabolic vertical velocity (positive to zero at apex)
            vz = self.takeoff_vz_peak * (1.0 - progress) - 2.0 * progress
            # Nose lowering toward neutral
            pitch_rate = -0.6 * np.sin(np.pi * progress)
            
            self.vel_world = np.array([vx, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 0.45-0.6: Distal initial contact (descent begins)
        elif phase < 0.6:
            progress = (phase - 0.45) / 0.15
            # Forward velocity begins to decrease
            vx = self.takeoff_vx_peak * 0.7 * (1.0 - 0.4 * progress)
            # Descending
            vz = -1.5 - 0.5 * progress
            # Small positive pitch for shallow reentry angle
            pitch_rate = 0.4 * np.sin(np.pi * progress / 2.0)
            
            self.vel_world = np.array([vx, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 0.6-0.8: Momentum dissipation
        elif phase < 0.8:
            progress = (phase - 0.6) / 0.2
            # Rapid forward velocity decay (friction from distal contact)
            vx = self.takeoff_vx_peak * 0.42 * (1.0 - progress) * np.exp(-3.0 * progress)
            # Controlled descent rate
            vz = -2.0 + 1.2 * progress
            # Nose lowering toward level
            pitch_rate = -0.5 * (1.0 - progress)
            
            self.vel_world = np.array([vx, 0.0, vz])
            self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Phase 0.8-1.0: Limb touchdown and stabilization
        else:
            progress = (phase - 0.8) / 0.2
            # All velocities decay to zero
            vx = self.takeoff_vx_peak * 0.05 * (1.0 - progress)**2
            vz = -0.8 * (1.0 - progress)**2
            # No angular velocity
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
        All legs move synchronously through:
        - Explosive extension (takeoff)
        - Aerial retraction
        - Distal-only contact (partial)
        - Full limb touchdown (stabilization)
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if front or rear leg for subtle forward/backward bias
        is_front = leg_name.startswith('F')
        front_bias = 0.02 if is_front else -0.02
        
        # Phase 0.0-0.2: Explosive takeoff (stance pushing)
        if phase < 0.2:
            progress = phase / 0.2
            # Rapid downward extension (pushes body up)
            foot[2] -= self.takeoff_extension * np.sin(np.pi * progress / 2.0)
            # Slight rearward motion during push
            foot[0] -= 0.03 * progress
        
        # Phase 0.2-0.45: Aerial ascent with retraction
        elif phase < 0.45:
            progress = (phase - 0.2) / 0.25
            # Retract toward body COM (upward and forward)
            foot[2] += self.retraction_height * np.sin(np.pi * progress / 2.0)
            # Position distal elements forward for reentry
            foot[0] += self.retraction_forward * progress + front_bias
            # Smooth transition from takeoff extension
            extension_decay = self.takeoff_extension * np.cos(np.pi * progress / 2.0)
            foot[2] -= extension_decay
        
        # Phase 0.45-0.8: Distal contact (partial, sliding)
        elif phase < 0.8:
            progress = (phase - 0.45) / 0.35
            # Main limb stays elevated (distal-only contact)
            foot[2] += self.distal_height_offset
            # Distal element positioned forward initially
            initial_forward = self.retraction_forward + front_bias
            # Rearward drift as momentum dissipates
            slide = self.distal_slide_distance * progress
            foot[0] += initial_forward - slide
            # Vertical oscillation representing tangential sliding dynamics
            foot[2] -= 0.03 * np.sin(2.0 * np.pi * progress)
        
        # Phase 0.8-1.0: Full limb touchdown and stabilization
        else:
            progress = (phase - 0.8) / 0.2
            # Transition from elevated distal position to full stance
            remaining_height = self.distal_height_offset * (1.0 - progress)
            foot[2] += remaining_height
            # Settle to neutral stance position in x
            end_x_offset = self.retraction_forward + front_bias - self.distal_slide_distance
            foot[0] += end_x_offset * (1.0 - progress)
            # Final settling (damped oscillation)
            foot[2] -= self.landing_settle_height * (1.0 - progress) * np.exp(-5.0 * progress)
        
        return foot