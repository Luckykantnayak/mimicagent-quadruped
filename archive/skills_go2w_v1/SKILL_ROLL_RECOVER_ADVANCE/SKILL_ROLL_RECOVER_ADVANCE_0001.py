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
        self.retraction_radius = 0.15  # Distance from body center during tuck
        self.retraction_height = -0.05  # Slight upward offset during tuck
        
        # Recovery parameters
        self.recovery_settle_velocity = -0.3  # Downward velocity during recovery

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
            # All angular rates remain zero
        
        # Phase 2: Roll execution [0.15, 0.5]
        elif phase < 0.5:
            vx = self.forward_velocity
            # Compute roll rate to complete 360 degrees over phase range
            roll_rate = self.roll_angle_total / self.roll_duration / (1.0 / self.freq)
        
        # Phase 3: Recovery extension [0.5, 0.65]
        elif phase < 0.65:
            # Decelerate forward velocity slightly
            recovery_progress = (phase - 0.5) / 0.15
            vx = self.forward_velocity * (1.0 - 0.2 * recovery_progress)
            vz = self.recovery_settle_velocity * (1.0 - recovery_progress)
            
            # Damp roll rate to zero
            roll_rate = 0.0  # Instant damping after roll completes
        
        # Phase 4: Stabilize advance [0.65, 1.0]
        else:
            vx = self.forward_velocity * 0.8  # Steady walking speed
            # All angular rates zero
        
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
            # Retract all legs toward body center
            roll_progress = (phase - 0.15) / 0.35
            
            # Smooth retraction using cosine interpolation
            retract_factor = 0.5 * (1.0 - np.cos(np.pi * min(roll_progress * 2.0, 1.0)))
            
            # Compute retracted position (close to body center)
            retracted_pos = np.array([
                nominal_pos[0] * (1.0 - retract_factor) + 0.0 * retract_factor,
                nominal_pos[1] * (1.0 - retract_factor) + 0.0 * retract_factor,
                self.retraction_height
            ])
            
            # Limit radial extent from body center
            radial_dist = np.sqrt(retracted_pos[0]**2 + retracted_pos[1]**2)
            if radial_dist > self.retraction_radius:
                scale = self.retraction_radius / radial_dist
                retracted_pos[0] *= scale
                retracted_pos[1] *= scale
            
            return retracted_pos
        
        # Phase 3: Recovery extension [0.5, 0.65]
        elif phase < 0.65:
            # Rapid extension back to nominal stance
            recovery_progress = (phase - 0.5) / 0.15
            
            # Smooth extension using sine interpolation for rapid deployment
            extend_factor = np.sin(np.pi * 0.5 * recovery_progress)
            
            # Compute current retracted position
            retracted_pos = np.array([0.0, 0.0, self.retraction_height])
            
            # Interpolate from retracted to nominal
            foot_pos = retracted_pos * (1.0 - extend_factor) + nominal_pos * extend_factor
            
            return foot_pos
        
        # Phase 4: Stabilize advance [0.65, 1.0]
        else:
            # Hold nominal stance position with minor settling
            stabilize_progress = (phase - 0.65) / 0.35
            settle_damping = 1.0 - 0.05 * np.exp(-5.0 * stabilize_progress)
            
            return nominal_pos * settle_damping