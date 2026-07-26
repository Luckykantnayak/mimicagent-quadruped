from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRAB_DIAGONAL_SCUTTLE_MotionGenerator(BaseMotionGenerator):
    """
    Crab diagonal scuttle gait with sideways body orientation.
    
    Motion consists of:
    - Two scuttle strokes (phase 0.0-0.3 and 0.5-0.8) with diagonal velocity
    - Rapid reset phase (0.3-0.5) for leg repositioning
    - Glide and stabilize phase (0.8-1.0) with velocity decay
    
    Front legs sweep rearward, rear legs sweep forward in body frame.
    Body maintains perpendicular orientation to travel direction via yaw rate.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Scuttle stroke parameters
        self.scuttle_sweep_amp_1 = 0.12  # First stroke amplitude
        self.scuttle_sweep_amp_2 = 0.16  # Second stroke amplitude (30% larger)
        self.reset_height = 0.06  # Height during rapid reset swing
        
        # Base velocity parameters
        self.vx_scuttle_1 = 0.8  # Forward velocity during first stroke
        self.vy_scuttle_1 = 0.6  # Lateral velocity during first stroke
        self.vx_scuttle_2 = 1.1  # Forward velocity during second stroke (higher)
        self.vy_scuttle_2 = 0.85  # Lateral velocity during second stroke (higher)
        self.yaw_rate_scuttle = 0.8  # Yaw rate to maintain perpendicular orientation
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion based on phase.
        
        Phase 0.0-0.3: First scuttle stroke with diagonal velocity and yaw
        Phase 0.3-0.5: Rapid reset, velocities near zero
        Phase 0.5-0.8: Second scuttle stroke with higher amplitude
        Phase 0.8-1.0: Glide with smooth velocity decay
        """
        
        if phase < 0.3:
            # First scuttle stroke: diagonal motion with yaw
            self.vel_world = np.array([self.vx_scuttle_1, self.vy_scuttle_1, 0.0])
            self.omega_world = np.array([0.0, 0.0, self.yaw_rate_scuttle])
            
        elif phase < 0.5:
            # Rapid reset: minimal base motion
            local_phase = (phase - 0.3) / 0.2
            # Small vertical oscillation during reset
            vz = 0.15 * np.sin(np.pi * local_phase)
            self.vel_world = np.array([0.0, 0.0, vz])
            self.omega_world = np.zeros(3)
            
        elif phase < 0.8:
            # Second scuttle stroke: higher amplitude diagonal motion
            self.vel_world = np.array([self.vx_scuttle_2, self.vy_scuttle_2, 0.0])
            self.omega_world = np.array([0.0, 0.0, self.yaw_rate_scuttle])
            
        else:
            # Glide and stabilize: smooth decay
            local_phase = (phase - 0.8) / 0.2
            decay_factor = np.cos(np.pi * local_phase / 2)  # Smooth decay from 1 to 0
            
            vx = self.vx_scuttle_2 * decay_factor
            vy = self.vy_scuttle_2 * decay_factor
            yaw_rate = self.yaw_rate_scuttle * decay_factor
            
            self.vel_world = np.array([vx, vy, 0.0])
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
        Compute foot position in body frame based on leg name and phase.
        
        Front legs (FL, FR): sweep rearward during scuttle strokes
        Rear legs (RL, RR): sweep forward during scuttle strokes
        All legs: rapid reset during phase 0.3-0.5
        All legs: hold near neutral during glide phase 0.8-1.0
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front_leg = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        if phase < 0.3:
            # First scuttle stroke
            local_phase = phase / 0.3
            sweep_progress = local_phase
            
            if is_front_leg:
                # Front legs sweep rearward (-x)
                foot[0] -= self.scuttle_sweep_amp_1 * sweep_progress
            else:
                # Rear legs sweep forward (+x)
                foot[0] += self.scuttle_sweep_amp_1 * sweep_progress
                
        elif phase < 0.5:
            # Rapid reset phase with swing
            local_phase = (phase - 0.3) / 0.2
            
            # Return to neutral position
            if is_front_leg:
                # Front legs return forward from rearward position
                foot[0] -= self.scuttle_sweep_amp_1 * (1.0 - local_phase)
            else:
                # Rear legs return rearward from forward position
                foot[0] += self.scuttle_sweep_amp_1 * (1.0 - local_phase)
            
            # Arc trajectory during reset
            swing_height = self.reset_height * np.sin(np.pi * local_phase)
            foot[2] += swing_height
            
        elif phase < 0.8:
            # Second scuttle stroke with larger amplitude
            local_phase = (phase - 0.5) / 0.3
            sweep_progress = local_phase
            
            if is_front_leg:
                # Front legs sweep rearward with larger amplitude
                foot[0] -= self.scuttle_sweep_amp_2 * sweep_progress
            else:
                # Rear legs sweep forward with larger amplitude
                foot[0] += self.scuttle_sweep_amp_2 * sweep_progress
                
        else:
            # Glide and stabilize: hold near neutral
            # Feet at end of second stroke, transitioning smoothly to neutral
            local_phase = (phase - 0.8) / 0.2
            decay_factor = 1.0 - local_phase
            
            if is_front_leg:
                foot[0] -= self.scuttle_sweep_amp_2 * decay_factor
            else:
                foot[0] += self.scuttle_sweep_amp_2 * decay_factor
        
        return foot