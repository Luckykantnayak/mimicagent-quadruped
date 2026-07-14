from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_TWIST_SHUFFLE_LATERAL_MotionGenerator(BaseMotionGenerator):
    """
    Twist-shuffle lateral motion with yaw oscillations and leg pushing.
    
    Motion cycle:
    - Phase 0.0-0.2: Base yaws CW to +30°, right legs (FR, RR) push laterally
    - Phase 0.2-0.4: Base returns to 0° yaw, left legs (FL, RL) extend
    - Phase 0.4-0.6: Base yaws CCW to -30°, left legs push laterally
    - Phase 0.6-0.8: Base returns to 0° yaw, right legs extend
    - Phase 0.8-1.0: Stabilization, all legs return to nominal
    
    All four feet remain in contact throughout (shuffle gait).
    Net result: leftward translation with zero net rotation.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slow cycle for coordinated shuffle
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.max_yaw_angle = np.radians(30.0)  # Maximum yaw excursion (30 degrees)
        self.lateral_extension = 0.12  # Lateral leg extension distance (m)
        self.lateral_velocity_magnitude = 0.3  # Lateral velocity during push phases (m/s)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with phase-dependent yaw oscillation and lateral velocity.
        """
        vx = 0.0
        vy = 0.0  # Lateral velocity (negative = leftward)
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.2: CW yaw to +30°, right legs push → leftward motion
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth yaw rate to reach +30° by end of phase
            yaw_rate = (self.max_yaw_angle / 0.2) * np.cos(np.pi * local_phase) * self.freq
            vy = -self.lateral_velocity_magnitude  # Leftward
            
        # Phase 0.2-0.4: Return yaw to 0°, left legs extend
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Negative yaw rate to return to neutral
            yaw_rate = -(self.max_yaw_angle / 0.2) * np.cos(np.pi * local_phase) * self.freq
            vy = -self.lateral_velocity_magnitude * 0.6  # Reduced but continued leftward
            
        # Phase 0.4-0.6: CCW yaw to -30°, left legs push → leftward motion
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Negative yaw rate to reach -30°
            yaw_rate = -(self.max_yaw_angle / 0.2) * np.cos(np.pi * local_phase) * self.freq
            vy = -self.lateral_velocity_magnitude  # Leftward
            
        # Phase 0.6-0.8: Return yaw to 0°, right legs extend
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Positive yaw rate to return to neutral
            yaw_rate = (self.max_yaw_angle / 0.2) * np.cos(np.pi * local_phase) * self.freq
            vy = -self.lateral_velocity_magnitude * 0.4  # Further reduced
            
        # Phase 0.8-1.0: Stabilization
        else:
            local_phase = (phase - 0.8) / 0.2
            # Smooth decay to zero
            yaw_rate = 0.0
            vy = -self.lateral_velocity_magnitude * 0.2 * (1.0 - local_phase)
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
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
        Compute foot position in body frame based on phase and leg group.
        All feet stay in contact; positions shift to create shuffle motion.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg group: right legs (FR, RR) vs left legs (FL, RL)
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        if is_right_leg:
            # Right legs: push phase 0.0-0.2, return 0.2-0.4, extend 0.6-0.8
            if phase < 0.2:
                # Push phase: extend outward (negative y) then move inward
                local_phase = phase / 0.2
                # Smooth extension and push
                extension = self.lateral_extension * np.sin(np.pi * local_phase)
                foot[1] -= extension  # Negative y = rightward in body frame
                
            elif phase < 0.4:
                # Return to nominal
                local_phase = (phase - 0.2) / 0.2
                extension = self.lateral_extension * (1.0 - local_phase)
                foot[1] -= extension
                
            elif phase < 0.6:
                # Support phase: maintain nominal position
                pass
                
            elif phase < 0.8:
                # Extend preparation for next cycle
                local_phase = (phase - 0.6) / 0.2
                extension = self.lateral_extension * local_phase
                foot[1] -= extension
                
            else:
                # Stabilization: return to nominal
                local_phase = (phase - 0.8) / 0.2
                extension = self.lateral_extension * (1.0 - local_phase)
                foot[1] -= extension
                
        elif is_left_leg:
            # Left legs: extend 0.2-0.4, push phase 0.4-0.6, return 0.6-0.8
            if phase < 0.2:
                # Support phase: maintain nominal position
                pass
                
            elif phase < 0.4:
                # Extend outward (positive y in body frame)
                local_phase = (phase - 0.2) / 0.2
                extension = self.lateral_extension * local_phase
                foot[1] += extension  # Positive y = leftward in body frame
                
            elif phase < 0.6:
                # Push phase: maintain extension then move inward
                local_phase = (phase - 0.4) / 0.2
                # Hold extension and create pushing motion
                extension = self.lateral_extension * (1.0 + 0.3 * np.sin(np.pi * local_phase))
                foot[1] += extension
                
            elif phase < 0.8:
                # Return to nominal
                local_phase = (phase - 0.6) / 0.2
                extension = self.lateral_extension * (1.0 - local_phase)
                foot[1] += extension
                
            else:
                # Stabilization: at nominal position
                pass
        
        return foot