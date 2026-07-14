from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SIDE_FLIP_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Side flip kinematic motion generator.
    
    Executes a 360° roll rotation while airborne with coordinated leg repositioning.
    
    Phase structure:
      [0.0, 0.25]: Launch and initial rotation
      [0.25, 0.5]: Inverted transition
      [0.5, 0.75]: Recovery rotation
      [0.75, 1.0]: Landing and stabilization
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # 2 seconds per full flip cycle
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Flip parameters
        self.peak_altitude = 0.6  # Maximum height during flip
        self.total_roll_rotation = 2 * np.pi  # 360 degrees
        
        # Velocity parameters
        self.launch_vz = 2.5  # Initial upward velocity
        self.gravity = -9.81  # Simulated gravity effect on kinematic trajectory
        
        # Roll rate tuned to complete 360° over the aerial phase
        self.peak_roll_rate = 8.0  # rad/s
        
        # Leg motion parameters
        self.leg_retract_height = 0.25  # How much legs retract during flip
        self.leg_lateral_swing = 0.15  # Lateral motion amplitude during rotation

    def update_base_motion(self, phase, dt):
        """
        Update base position and orientation through flip phases.
        """
        
        # Phase-dependent vertical velocity (ballistic trajectory)
        if phase < 0.15:
            # Launch phase: strong upward velocity
            vz = self.launch_vz * (1.0 - phase / 0.15)
        elif phase < 0.5:
            # Ascending to apex and beginning descent
            apex_phase = (phase - 0.15) / 0.35
            vz = self.launch_vz * (1.0 - apex_phase) + self.gravity * apex_phase * 0.5
        elif phase < 0.8:
            # Descent phase
            descent_phase = (phase - 0.5) / 0.3
            vz = self.gravity * 0.5 + self.gravity * descent_phase * 0.5
        else:
            # Landing deceleration
            land_phase = (phase - 0.8) / 0.2
            vz = self.gravity * (1.0 - land_phase * 0.9)
        
        # Roll rate profile: high during aerial phase, zero at launch/landing
        if phase < 0.05:
            # Initial roll rate buildup
            roll_rate = self.peak_roll_rate * (phase / 0.05)
        elif phase < 0.85:
            # Sustained roll during aerial phase
            roll_rate = self.peak_roll_rate
        else:
            # Roll rate decay for landing
            land_phase = (phase - 0.85) / 0.15
            roll_rate = self.peak_roll_rate * (1.0 - land_phase)
        
        # Set velocity commands
        self.vel_world = np.array([0.0, 0.0, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
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
        Compute foot position in body frame throughout flip.
        
        Legs reposition through coordinated arcs:
        - Launch: extended downward
        - Aerial/inverted: retracted and repositioned overhead
        - Recovery: transition back to downward
        - Landing: extended to nominal stance
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine leg side for symmetric motion
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        lateral_sign = 1.0 if is_left else -1.0
        
        # Phase-dependent foot trajectory
        if phase < 0.15:
            # Launch phase: feet on ground, slightly extended
            foot[2] = base_pos[2] - 0.02  # Slight downward extension
            
        elif phase < 0.5:
            # Aerial phase: retract and swing overhead as body rotates
            aerial_progress = (phase - 0.15) / 0.35
            
            # Retract upward in body frame (which is rotating)
            foot[2] = base_pos[2] + self.leg_retract_height * np.sin(np.pi * aerial_progress)
            
            # Lateral swing to track rotation
            foot[1] = base_pos[1] + lateral_sign * self.leg_lateral_swing * np.sin(np.pi * aerial_progress)
            
            # Slight forward/back adjustment
            if is_front:
                foot[0] = base_pos[0] + 0.05 * np.sin(np.pi * aerial_progress)
            else:
                foot[0] = base_pos[0] - 0.05 * np.sin(np.pi * aerial_progress)
                
        elif phase < 0.85:
            # Recovery phase: transition from overhead back to downward
            recovery_progress = (phase - 0.5) / 0.35
            
            # Return from retracted to extended
            foot[2] = base_pos[2] + self.leg_retract_height * np.sin(np.pi * (1.0 - recovery_progress))
            
            # Return lateral position
            foot[1] = base_pos[1] + lateral_sign * self.leg_lateral_swing * np.sin(np.pi * (1.0 - recovery_progress))
            
            # Return longitudinal position
            if is_front:
                foot[0] = base_pos[0] + 0.05 * np.sin(np.pi * (1.0 - recovery_progress))
            else:
                foot[0] = base_pos[0] - 0.05 * np.sin(np.pi * (1.0 - recovery_progress))
                
        else:
            # Landing phase: extend to nominal stance with slight damping motion
            land_progress = (phase - 0.85) / 0.15
            
            # Smooth extension to ground contact
            foot[2] = base_pos[2] - 0.05 * land_progress
            
            # Return to base lateral position
            foot[1] = base_pos[1]
            foot[0] = base_pos[0]
        
        return foot