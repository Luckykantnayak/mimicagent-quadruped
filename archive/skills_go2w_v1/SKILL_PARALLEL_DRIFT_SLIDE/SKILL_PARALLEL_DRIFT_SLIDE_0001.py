from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_PARALLEL_DRIFT_SLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Parallel drift slide motion generator.
    
    The robot slides diagonally forward-left while its body is rotated 45 degrees 
    to the right of the travel direction, resembling a car drift.
    
    - Left legs (FL, RL) perform alternating active lateral push-slides perpendicular 
      to body axis to generate lateral thrust
    - Right legs (FR, RR) maintain passive dragging contact
    - Body maintains constant height and fixed 45-degree yaw offset throughout
    - All four feet remain in continuous ground contact (sliding motion)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower phase evolution for controlled drift
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Drift motion parameters
        self.lateral_slide_amplitude = 0.06  # Lateral push distance for left legs (meters)
        self.drag_offset_x = -0.02  # Slight rearward offset for right legs
        self.drag_offset_y = -0.01  # Slight inward offset for right legs
        
        # Base velocity parameters (BODY frame)
        # When body is yawed 45 deg, equal vx and vy produce diagonal world trajectory
        self.vx_body = 0.3  # Forward velocity in body frame (m/s)
        self.vy_body = 0.3  # Leftward velocity in body frame (m/s)
        
        # Initial 45-degree yaw offset (right of travel direction)
        self.initial_yaw_offset = np.pi / 4  # 45 degrees in radians
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        # Initialize with 45-degree yaw offset
        self.root_quat = euler_to_quat(0.0, 0.0, self.initial_yaw_offset)
        
        # Phase offsets for alternating push cycles
        # FL pushes during [0.0, 0.5], RL pushes during [0.5, 1.0]
        self.phase_offsets = {
            leg_names[0]: 0.0,  # FL
            leg_names[1]: 0.0,  # FR
            leg_names[2]: 0.5,  # RL (180 degrees out of phase with FL)
            leg_names[3]: 0.0,  # RR
        }

    def update_base_motion(self, phase, dt):
        """
        Update base with constant diagonal velocity and zero yaw rate.
        
        Body-frame velocities vx and vy combine to produce diagonal world-frame 
        translation when body is yawed 45 degrees. Zero yaw rate maintains 
        constant heading offset throughout the drift.
        """
        # Constant body-frame velocities for diagonal drift
        self.vel_world = np.array([self.vx_body, self.vy_body, 0.0])
        
        # Zero angular velocity to maintain fixed 45-degree yaw offset
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
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
        Compute foot position in BODY frame for each leg.
        
        FL and RL: Alternating active lateral push-slides (perpendicular to body axis)
        FR and RR: Passive dragging contact with minimal motion
        """
        # Get base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Left legs (FL, RL): Active lateral push-slide motion
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            if leg_phase < 0.5:
                # Active push phase: slide from +y (left) toward centerline
                # Smooth sinusoidal trajectory for force modulation
                progress = leg_phase / 0.5  # Normalize to [0, 1]
                lateral_offset = self.lateral_slide_amplitude * np.cos(np.pi * progress)
                foot[1] += lateral_offset  # Positive y is left in body frame
            else:
                # Recovery phase: gently return from centerline back to +y (left)
                progress = (leg_phase - 0.5) / 0.5  # Normalize to [0, 1]
                lateral_offset = -self.lateral_slide_amplitude * np.cos(np.pi * progress)
                foot[1] += lateral_offset
        
        # Right legs (FR, RR): Passive drag with minimal motion
        elif leg_name.startswith('FR'):
            # Front-right: passive drag with slight rearward and inward offset
            foot[0] += self.drag_offset_x  # Slight rearward
            foot[1] += self.drag_offset_y  # Slight inward
            # Slow rearward drift during RL push phase to avoid interference
            if leg_phase >= 0.5:
                drift_progress = (leg_phase - 0.5) / 0.5
                foot[0] += -0.01 * drift_progress  # Small additional rearward drift
        
        elif leg_name.startswith('RR'):
            # Rear-right: continuous passive drag throughout full cycle
            foot[0] += self.drag_offset_x  # Slight rearward
            foot[1] += self.drag_offset_y  # Slight inward
            # Minimal sinusoidal variation to simulate continuous drag adaptation
            foot[0] += -0.005 * np.sin(2 * np.pi * leg_phase)
        
        return foot