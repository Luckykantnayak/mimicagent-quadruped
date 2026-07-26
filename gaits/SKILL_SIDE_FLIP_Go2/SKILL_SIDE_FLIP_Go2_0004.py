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
        self.nominal_height = 0.28  # Standing height
        self.peak_altitude = 0.58  # Maximum height during flip (below 0.65 m limit)
        self.total_roll_rotation = 2 * np.pi  # 360 degrees
        
        # Roll rate tuned to complete 360° over the aerial phase
        self.peak_roll_rate = 7.5  # rad/s
        
        # Leg motion parameters
        self.leg_retract_height = 0.20  # How much legs retract during flip
        self.leg_lateral_swing = 0.12  # Lateral motion amplitude during rotation

    def compute_height_trajectory(self, phase):
        """
        Compute explicit height trajectory as function of phase.
        Returns height above ground and vertical velocity.
        """
        if phase < 0.2:
            # Launch phase: smooth rise from nominal to peak
            t = phase / 0.2
            # Smooth cubic interpolation
            h = self.nominal_height + (self.peak_altitude - self.nominal_height) * (3 * t**2 - 2 * t**3)
            # Velocity is derivative
            v = (self.peak_altitude - self.nominal_height) * (6 * t - 6 * t**2) / 0.2
        elif phase < 0.7:
            # Sustained altitude during main rotation
            t = (phase - 0.2) / 0.5
            # Slight sinusoidal variation for smoothness
            h = self.peak_altitude - 0.03 * np.sin(np.pi * t)
            v = -0.03 * np.pi * np.cos(np.pi * t) / 0.5
        else:
            # Landing phase: smooth descent back to nominal
            t = (phase - 0.7) / 0.3
            # Smooth cubic interpolation
            h = self.peak_altitude - (self.peak_altitude - self.nominal_height) * (3 * t**2 - 2 * t**3)
            # Velocity is derivative
            v = -(self.peak_altitude - self.nominal_height) * (6 * t - 6 * t**2) / 0.3
        
        return h, v

    def update_base_motion(self, phase, dt):
        """
        Update base position and orientation through flip phases.
        """
        
        # Compute explicit height trajectory
        target_height, vz = self.compute_height_trajectory(phase)
        
        # Set vertical position directly to ensure bounded trajectory
        self.root_pos[2] = target_height
        
        # Roll rate profile: smooth ramp up, sustained, smooth ramp down
        if phase < 0.08:
            # Initial roll rate buildup with smooth cubic
            t = phase / 0.08
            roll_rate = self.peak_roll_rate * (3 * t**2 - 2 * t**3)
        elif phase < 0.82:
            # Sustained roll during aerial phase
            roll_rate = self.peak_roll_rate
        else:
            # Roll rate decay for landing with smooth cubic
            t = (phase - 0.82) / 0.18
            roll_rate = self.peak_roll_rate * (1 - (3 * t**2 - 2 * t**3))
        
        # Set velocity commands
        self.vel_world = np.array([0.0, 0.0, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
        # Integrate orientation only (position set directly above)
        # Use a temporary integration for quaternion
        _, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            np.array([0.0, 0.0, 0.0]),  # No position integration
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame throughout flip.
        
        Legs reposition through coordinated arcs:
        - Launch: extended downward
        - Aerial/inverted: retracted with adaptive compensation
        - Recovery: transition back to downward
        - Landing: extended to nominal stance
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine leg side for symmetric motion
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        lateral_sign = 1.0 if is_left else -1.0
        
        # Compute current roll angle to modulate retraction during inversion
        # Roll progresses approximately linearly with phase during main rotation
        estimated_roll = self.total_roll_rotation * np.clip((phase - 0.05) / 0.85, 0.0, 1.0)
        
        # Inversion factor: 1.0 when upright, 0.0 when inverted (roll ~ pi)
        inversion_factor = 0.5 + 0.5 * np.cos(estimated_roll)
        
        # Phase-dependent foot trajectory with smooth transitions
        if phase < 0.15:
            # Launch phase: feet in nominal stance with smooth transition
            t = phase / 0.15
            smooth_t = 3 * t**2 - 2 * t**3
            foot[2] = base_pos[2] - 0.01 * smooth_t
            
        elif phase < 0.25:
            # Early aerial: begin retraction
            t = (phase - 0.15) / 0.1
            smooth_t = 3 * t**2 - 2 * t**3
            
            # Smooth retraction with modulation
            retract_amount = self.leg_retract_height * smooth_t * (0.6 + 0.4 * inversion_factor)
            foot[2] = base_pos[2] - 0.01 + retract_amount
            
            # Begin lateral swing
            foot[1] = base_pos[1] + lateral_sign * self.leg_lateral_swing * smooth_t * 0.5
            
        elif phase < 0.6:
            # Main aerial/inverted phase: sustained retraction with adaptive height
            t = (phase - 0.25) / 0.35
            
            # Adaptive retraction: reduce when inverted to prevent ground contact
            retract_amount = self.leg_retract_height * (0.6 + 0.4 * inversion_factor)
            foot[2] = base_pos[2] + retract_amount
            
            # Lateral swing with smooth sinusoidal profile
            foot[1] = base_pos[1] + lateral_sign * self.leg_lateral_swing * (0.5 + 0.5 * np.sin(np.pi * (t - 0.5)))
            
            # Longitudinal adjustment for balance
            long_offset = 0.04 * np.sin(np.pi * t)
            if is_front:
                foot[0] = base_pos[0] + long_offset
            else:
                foot[0] = base_pos[0] - long_offset
                
        elif phase < 0.78:
            # Recovery phase: transition from retracted to extended
            t = (phase - 0.6) / 0.18
            smooth_t = 3 * t**2 - 2 * t**3
            
            # Smooth return from retraction
            retract_amount = self.leg_retract_height * (0.6 + 0.4 * inversion_factor) * (1.0 - smooth_t)
            foot[2] = base_pos[2] + retract_amount
            
            # Return lateral position
            lateral_amount = self.leg_lateral_swing * (0.5 + 0.5 * np.sin(np.pi * (0.5 + 0.35 * (1.0 - smooth_t))))
            foot[1] = base_pos[1] + lateral_sign * lateral_amount * (1.0 - smooth_t)
            
            # Return longitudinal position
            long_offset = 0.04 * np.sin(np.pi * (1.0 - smooth_t))
            if is_front:
                foot[0] = base_pos[0] + long_offset
            else:
                foot[0] = base_pos[0] - long_offset
                
        else:
            # Landing phase: smooth extension to nominal stance
            t = (phase - 0.78) / 0.22
            smooth_t = 3 * t**2 - 2 * t**3
            
            # Smooth extension with slight damping
            foot[2] = base_pos[2] - 0.02 * smooth_t
            
            # Return to base position
            foot[1] = base_pos[1]
            foot[0] = base_pos[0]
        
        return foot