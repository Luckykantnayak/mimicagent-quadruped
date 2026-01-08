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
        
        # Leg retraction parameters (body frame)
        self.retraction_radius = 0.22  # Increased to reduce joint stress
        self.retraction_height = -0.28  # Below body to keep feet tucked downward
        
        # Base lift parameters for ground clearance during roll
        self.max_lift_height = 0.25  # Maximum base elevation during roll
        
        # Recovery parameters
        self.recovery_settle_velocity = -0.25  # Downward velocity during late recovery

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
            # Begin gentle ascent at end of pre-roll
            if phase > 0.10:
                pre_lift_progress = (phase - 0.10) / 0.05
                vz = self.max_lift_height * 2.0 * pre_lift_progress
        
        # Phase 2: Roll execution [0.15, 0.5]
        elif phase < 0.5:
            vx = self.forward_velocity
            # Compute roll rate to complete 360 degrees over phase range
            roll_rate = self.roll_angle_total / self.roll_duration / (1.0 / self.freq)
            
            # Maintain elevation during roll with sinusoidal profile
            # Lift peaks at 90 and 270 degrees (phase 0.2375 and 0.4125)
            roll_progress = (phase - 0.15) / 0.35
            roll_angle = roll_progress * self.roll_angle_total
            
            # Compute desired elevation as function of roll angle
            # Maximum at 90 and 270 degrees, minimum at 0, 180, 360
            elevation_factor = abs(np.sin(roll_angle))
            target_vz = self.max_lift_height * 1.5  # Maintain height
            
            vz = target_vz * elevation_factor * 0.5
        
        # Phase 3: Recovery extension [0.5, 0.65]
        elif phase < 0.65:
            # Decelerate forward velocity slightly
            recovery_progress = (phase - 0.5) / 0.15
            vx = self.forward_velocity * (1.0 - 0.15 * recovery_progress)
            
            # Delayed descent: only begin descending after legs partially extend
            if recovery_progress > 0.4:
                descent_progress = (recovery_progress - 0.4) / 0.6
                vz = self.recovery_settle_velocity * descent_progress
            else:
                vz = 0.0
            
            # Roll rate instantly damped
            roll_rate = 0.0
        
        # Phase 4: Stabilize advance [0.65, 1.0]
        else:
            vx = self.forward_velocity * 0.85  # Steady walking speed
            # Gentle settling
            stabilize_progress = (phase - 0.65) / 0.35
            if stabilize_progress < 0.3:
                vz = self.recovery_settle_velocity * 0.3 * (1.0 - stabilize_progress / 0.3)
        
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
            # Retract all legs toward body center with multi-stage trajectory
            roll_progress = (phase - 0.15) / 0.35
            
            # Smooth retraction using cosine interpolation
            retract_factor = 0.5 * (1.0 - np.cos(np.pi * roll_progress))
            
            # Multi-stage retraction: prioritize vertical lift early, then horizontal convergence
            if roll_progress < 0.3:
                # Early stage: lift feet vertically while maintaining lateral position
                vertical_progress = roll_progress / 0.3
                vertical_factor = 0.5 * (1.0 - np.cos(np.pi * vertical_progress))
                
                retracted_pos = np.array([
                    nominal_pos[0] * (1.0 - 0.2 * vertical_factor),
                    nominal_pos[1] * (1.0 - 0.2 * vertical_factor),
                    nominal_pos[2] * (1.0 - vertical_factor) + self.retraction_height * vertical_factor
                ])
            else:
                # Late stage: converge horizontally toward body center
                horizontal_progress = (roll_progress - 0.3) / 0.7
                horizontal_factor = 0.5 * (1.0 - np.cos(np.pi * horizontal_progress))
                
                # Intermediate position from early stage
                early_pos = np.array([
                    nominal_pos[0] * 0.8,
                    nominal_pos[1] * 0.8,
                    self.retraction_height
                ])
                
                # Target retracted position
                target_radial = self.retraction_radius
                radial_nominal = np.sqrt(nominal_pos[0]**2 + nominal_pos[1]**2)
                
                if radial_nominal > 0:
                    scale = target_radial / radial_nominal
                else:
                    scale = 1.0
                
                target_pos = np.array([
                    nominal_pos[0] * scale,
                    nominal_pos[1] * scale,
                    self.retraction_height
                ])
                
                retracted_pos = early_pos * (1.0 - horizontal_factor) + target_pos * horizontal_factor
            
            return retracted_pos
        
        # Phase 3: Recovery extension [0.5, 0.65]
        elif phase < 0.65:
            # Rapid extension back to nominal stance
            recovery_progress = (phase - 0.5) / 0.15
            
            # Smooth extension using sine interpolation for rapid deployment
            extend_factor = np.sin(np.pi * 0.5 * recovery_progress)
            
            # Compute current retracted position (final configuration from roll phase)
            radial_nominal = np.sqrt(nominal_pos[0]**2 + nominal_pos[1]**2)
            if radial_nominal > 0:
                scale = self.retraction_radius / radial_nominal
            else:
                scale = 1.0
            
            retracted_pos = np.array([
                nominal_pos[0] * scale,
                nominal_pos[1] * scale,
                self.retraction_height
            ])
            
            # Interpolate from retracted to nominal
            foot_pos = retracted_pos * (1.0 - extend_factor) + nominal_pos * extend_factor
            
            return foot_pos
        
        # Phase 4: Stabilize advance [0.65, 1.0]
        else:
            # Hold nominal stance position
            return nominal_pos