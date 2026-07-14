from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ROLL_RECOVER_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Execute a complete 360-degree longitudinal roll while maintaining forward momentum.
    
    Phase structure:
    - [0.0, 0.15]: Pre-roll advance with stable quadrupedal stance
    - [0.15, 0.5]: Roll execution with all legs tucked, 360-degree rotation about longitudinal axis
    - [0.5, 0.65]: Recovery extension with rapid leg deployment and angular damping
    - [0.65, 1.0]: Stabilize advance with restored stance and forward locomotion
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower cycle to allow smooth roll execution
        
        # Store nominal stance positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.forward_velocity = 0.8  # Sustained forward velocity throughout
        self.roll_duration = 0.35  # Phase range 0.15-0.5
        self.roll_angle_total = 2 * np.pi  # 360 degrees
        
        # Leg retraction parameters - conservative to respect joint limits
        self.retraction_scale = 0.82  # Moderate scaling toward body center
        self.retraction_height_offset = -0.06  # Minimal downward offset
        
        # Base lift parameters for ground clearance during roll
        self.max_lift_height = 0.50  # Increased elevation for clearance during inversion
        
        # Recovery parameters - no descent until legs fully extended
        self.recovery_settle_velocity = -0.20  # Gentle downward velocity in late stabilization

    def update_base_motion(self, phase, dt):
        """
        Update base pose based on phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 1: Pre-roll advance [0.0, 0.15]
        if phase < 0.15:
            vx = self.forward_velocity
            # Begin gentle ascent toward end of pre-roll
            if phase > 0.08:
                pre_lift_progress = (phase - 0.08) / 0.07
                # Smooth lift initiation
                lift_factor = 0.5 * (1.0 - np.cos(np.pi * pre_lift_progress))
                vz = self.max_lift_height * 2.5 * lift_factor
        
        # Phase 2: Roll execution [0.15, 0.5]
        elif phase < 0.5:
            vx = self.forward_velocity
            # Compute roll rate to complete 360 degrees over phase range
            roll_rate = self.roll_angle_total / self.roll_duration / (1.0 / self.freq)
            
            # Maintain sustained elevation throughout roll
            roll_progress = (phase - 0.15) / 0.35
            
            # Early roll: continue ascending to maximum height
            if roll_progress < 0.20:
                ascent_progress = roll_progress / 0.20
                ascent_ease = 0.5 * (1.0 - np.cos(np.pi * ascent_progress))
                vz = self.max_lift_height * 1.8 * (1.0 - ascent_ease)
            # Mid and late roll: maintain constant elevation
            else:
                vz = 0.0
        
        # Phase 3: Recovery extension [0.5, 0.65]
        elif phase < 0.65:
            # Maintain forward velocity with slight deceleration
            recovery_progress = (phase - 0.5) / 0.15
            vx = self.forward_velocity * (1.0 - 0.08 * recovery_progress)
            
            # Maintain constant height during leg extension - no descent
            vz = 0.0
            
            # Roll rate damped to zero
            roll_rate = 0.0
        
        # Phase 4: Stabilize advance [0.65, 1.0]
        else:
            vx = self.forward_velocity * 0.85  # Steady walking speed
            # Delayed gentle settling after legs are fully extended
            stabilize_progress = (phase - 0.65) / 0.35
            if stabilize_progress > 0.30:  # Begin descent only after 0.75 phase
                settle_progress = (stabilize_progress - 0.30) / 0.70
                settle_ease = 0.5 * (1.0 - np.cos(np.pi * settle_progress))
                vz = self.recovery_settle_velocity * settle_ease
            else:
                vz = 0.0
        
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
        Compute foot position in body frame based on phase and leg.
        """
        nominal_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Phase 1: Pre-roll advance [0.0, 0.15]
        if phase < 0.15:
            # Maintain nominal stance position
            return nominal_pos
        
        # Phase 2: Roll execution [0.15, 0.5]
        elif phase < 0.5:
            # Conservative retraction: scale toward body center with minimal vertical change
            roll_progress = (phase - 0.15) / 0.35
            
            # Smooth retraction using smoothstep easing
            retract_factor = roll_progress * roll_progress * (3.0 - 2.0 * roll_progress)
            
            # Compute retracted position with conservative scaling
            retracted_x = nominal_pos[0] * self.retraction_scale
            retracted_y = nominal_pos[1] * self.retraction_scale
            
            # Minimal vertical offset to avoid excessive hip joint demands
            retracted_z = nominal_pos[2] + self.retraction_height_offset * retract_factor
            
            # Interpolate from nominal to retracted
            foot_pos = np.array([
                nominal_pos[0] * (1.0 - retract_factor) + retracted_x * retract_factor,
                nominal_pos[1] * (1.0 - retract_factor) + retracted_y * retract_factor,
                nominal_pos[2] * (1.0 - retract_factor) + retracted_z * retract_factor
            ])
            
            return foot_pos
        
        # Phase 3: Recovery extension [0.5, 0.65]
        elif phase < 0.65:
            # Smooth extension back to nominal stance
            recovery_progress = (phase - 0.5) / 0.15
            
            # Use smoothstep easing for controlled deployment
            extend_factor = recovery_progress * recovery_progress * (3.0 - 2.0 * recovery_progress)
            
            # Compute fully retracted position (end state from roll phase)
            retracted_pos = np.array([
                nominal_pos[0] * self.retraction_scale,
                nominal_pos[1] * self.retraction_scale,
                nominal_pos[2] + self.retraction_height_offset
            ])
            
            # Interpolate from retracted to nominal
            foot_pos = retracted_pos * (1.0 - extend_factor) + nominal_pos * extend_factor
            
            return foot_pos
        
        # Phase 4: Stabilize advance [0.65, 1.0]
        else:
            # Hold nominal stance position
            return nominal_pos