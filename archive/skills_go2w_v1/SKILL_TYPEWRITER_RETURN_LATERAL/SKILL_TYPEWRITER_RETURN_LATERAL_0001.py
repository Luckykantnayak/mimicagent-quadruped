from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_TYPEWRITER_RETURN_LATERAL_MotionGenerator(BaseMotionGenerator):
    """
    Typewriter carriage return lateral motion: cyclic sliding with drift-snap rhythm.
    
    Motion characteristics:
    - All four legs maintain continuous ground contact (pure sliding, no flight phase)
    - Drift phases (0-0.4, 0.45-0.85): smooth rightward base velocity, feet slide left in body frame
    - Return phases (0.4-0.45, 0.85-0.9): sharp leftward base velocity, feet snap right in body frame
    - Settle phase (0.9-1.0): velocities ramp down for clean cycle transition
    - Net motion is rightward with periodic rapid leftward corrections
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Lateral sliding parameters (body frame y-axis)
        self.drift_slide_distance = 0.12  # Total lateral slide distance during drift phase
        self.return_slide_distance = 0.08  # Rapid repositioning distance during return
        
        # Base velocity parameters
        self.drift_velocity = 0.15  # Moderate rightward velocity during drift (m/s)
        self.return_velocity = -0.6  # Sharp leftward velocity during return (m/s)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent lateral velocity commands.
        Creates typewriter-like drift and snap pattern.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        # First drift phase: smooth rightward motion
        if 0.0 <= phase < 0.4:
            vy = self.drift_velocity
        
        # First return phase: rapid leftward snap
        elif 0.4 <= phase < 0.45:
            vy = self.return_velocity
        
        # Second drift phase: smooth rightward motion
        elif 0.45 <= phase < 0.85:
            vy = self.drift_velocity
        
        # Second return phase: rapid leftward snap
        elif 0.85 <= phase < 0.9:
            vy = self.return_velocity
        
        # Settle phase: smooth velocity reduction
        elif 0.9 <= phase <= 1.0:
            # Linear ramp down from previous velocity to zero
            settle_progress = (phase - 0.9) / 0.1
            vy = self.return_velocity * (1.0 - settle_progress)
        
        # Set velocity commands (world frame)
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame during sliding motion.
        
        All legs move synchronously:
        - During drift: feet slide leftward in body frame (as base moves right)
        - During return: feet snap rightward in body frame (rapid repositioning)
        - During settle: feet stabilize at nominal position
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Lateral offset in body frame (y-axis)
        lateral_offset = 0.0
        
        # First drift phase (0.0 - 0.4): smooth leftward slide in body frame
        if 0.0 <= phase < 0.4:
            drift_progress = phase / 0.4
            lateral_offset = -self.drift_slide_distance * drift_progress
        
        # First return phase (0.4 - 0.45): rapid rightward snap
        elif 0.4 <= phase < 0.45:
            return_progress = (phase - 0.4) / 0.05
            # Start from end of drift, snap rightward
            lateral_offset = -self.drift_slide_distance + self.return_slide_distance * return_progress
        
        # Second drift phase (0.45 - 0.85): smooth leftward slide from reset position
        elif 0.45 <= phase < 0.85:
            drift_progress = (phase - 0.45) / 0.4
            # Start from position after first return
            start_offset = -self.drift_slide_distance + self.return_slide_distance
            lateral_offset = start_offset - self.drift_slide_distance * drift_progress
        
        # Second return phase (0.85 - 0.9): rapid rightward snap
        elif 0.85 <= phase < 0.9:
            return_progress = (phase - 0.85) / 0.05
            # Start from end of second drift, snap rightward
            start_offset = -self.drift_slide_distance + self.return_slide_distance - self.drift_slide_distance
            lateral_offset = start_offset + self.return_slide_distance * return_progress
        
        # Settle phase (0.9 - 1.0): smooth return to nominal position
        elif 0.9 <= phase <= 1.0:
            settle_progress = (phase - 0.9) / 0.1
            # Current position at phase 0.9
            current_offset = -self.drift_slide_distance + self.return_slide_distance - self.drift_slide_distance + self.return_slide_distance
            # Smoothly interpolate back to zero for clean cycle restart
            lateral_offset = current_offset * (1.0 - settle_progress)
        
        # Apply lateral offset to foot position (body frame y-axis)
        foot[1] += lateral_offset
        
        # Maintain ground contact: z position remains at base height
        # (no vertical oscillation, pure sliding motion)
        
        return foot