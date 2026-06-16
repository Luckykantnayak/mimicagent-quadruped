from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SIDE_FLIP_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Side flip maneuver: 360-degree roll rotation with aerial phase.
    
    Phase structure:
      [0.0, 0.1]   launch_prep: all legs grounded, generate upward velocity and roll rate
      [0.1, 0.4]   aerial_rotation_ascent: legs off ground, roll 0° → ~180°, ascending
      [0.4, 0.6]   aerial_rotation_inverted: pass through inverted, roll ~180°
      [0.6, 0.85]  aerial_rotation_descent: roll 180° → 360°, descending
      [0.85, 1.0]  landing_and_recovery: all legs re-establish contact, velocities → 0
    
    Base motion: kinematic integration of vertical velocity and roll rate
    Leg motion: reposition continuously in body frame to track rotating body
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slow cycle for dramatic aerial flip
        
        # Base foot positions (body frame, nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.launch_vz = 2.5  # Upward launch velocity (m/s)
        self.roll_rate = 2.0 * np.pi * 0.8  # ~288 deg/s to achieve 360° rotation during aerial phase
        self.apex_height = 0.6  # Peak height above ground during flip
        self.leg_tuck_height = 0.15  # How much to retract legs during aerial phase
        self.leg_lateral_spread = 0.08  # Lateral repositioning during rotation

    def update_base_motion(self, phase, dt):
        """
        Integrate base pose using phase-dependent velocity commands.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.1: launch preparation
        if phase < 0.1:
            progress = phase / 0.1
            # Ramp up vertical velocity and roll rate
            vz = self.launch_vz * progress
            roll_rate = self.roll_rate * progress
        
        # Phase 0.1-0.4: aerial ascent, continuous roll
        elif phase < 0.4:
            progress = (phase - 0.1) / 0.3
            # Vertical velocity decreases (gravity effect kinematically modeled)
            vz = self.launch_vz * (1.0 - progress * 1.5)
            roll_rate = self.roll_rate
        
        # Phase 0.4-0.6: inverted phase, apex transition
        elif phase < 0.6:
            progress = (phase - 0.4) / 0.2
            # Transition from upward to downward velocity (apex at midpoint)
            vz = self.launch_vz * 0.5 * (1.0 - progress) - self.launch_vz * 0.5 * progress
            roll_rate = self.roll_rate
        
        # Phase 0.6-0.85: aerial descent, continue roll
        elif phase < 0.85:
            progress = (phase - 0.6) / 0.25
            # Downward velocity increases (falling)
            vz = -self.launch_vz * 0.5 * (1.0 + progress)
            roll_rate = self.roll_rate
        
        # Phase 0.85-1.0: landing and recovery
        else:
            progress = (phase - 0.85) / 0.15
            # Decelerate vertical velocity to zero
            vz = -self.launch_vz * (1.0 - progress)
            # Decelerate roll rate to zero
            roll_rate = self.roll_rate * (1.0 - progress)
        
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

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame throughout the flip.
        Legs reposition to maintain kinematic validity as body rotates.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine leg side (left vs right) for symmetric repositioning
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        lateral_sign = 1.0 if is_left else -1.0
        
        # Phase 0.0-0.1: launch prep, legs grounded at nominal stance
        if phase < 0.1:
            foot = base_pos.copy()
        
        # Phase 0.1-0.4: aerial ascent, retract legs and begin lateral repositioning
        elif phase < 0.4:
            progress = (phase - 0.1) / 0.3
            # Retract legs upward in body frame (reduce -z, increase z)
            foot[2] += self.leg_tuck_height * progress
            # Move legs laterally to clear rotation
            foot[1] += lateral_sign * self.leg_lateral_spread * progress
        
        # Phase 0.4-0.6: inverted phase, legs tucked and repositioned
        elif phase < 0.6:
            progress = (phase - 0.4) / 0.2
            # Maintain tucked position, transition lateral positioning
            foot[2] += self.leg_tuck_height
            # Transition from one lateral extreme to opposite to track body rotation
            foot[1] += lateral_sign * self.leg_lateral_spread * (1.0 - 2.0 * progress)
        
        # Phase 0.6-0.85: aerial descent, extend legs back toward stance
        elif phase < 0.85:
            progress = (phase - 0.6) / 0.25
            # Extend legs back downward in body frame
            foot[2] += self.leg_tuck_height * (1.0 - progress)
            # Return legs to nominal lateral position
            foot[1] += lateral_sign * self.leg_lateral_spread * (progress - 1.0)
        
        # Phase 0.85-1.0: landing, legs fully extended to nominal stance
        else:
            progress = (phase - 0.85) / 0.15
            # Smoothly return to base stance position
            foot = base_pos.copy()
        
        return foot