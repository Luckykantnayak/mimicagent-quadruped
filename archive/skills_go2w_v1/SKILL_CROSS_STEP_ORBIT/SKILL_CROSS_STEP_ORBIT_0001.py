from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_CROSS_STEP_ORBIT_MotionGenerator(BaseMotionGenerator):
    """
    Crossover stepping orbit motion with continuous lateral velocity and yaw rotation.
    
    - Outer legs (FL, RL) and inner legs (FR, RR) alternate between crossing and uncrossing
    - Base moves laterally rightward (body +y) with continuous yaw rotation
    - Diagonal support pattern maintained throughout for stability
    - Produces circular trajectory in world frame with body tangent to orbit
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for complex crossover motion
        
        # Crossover motion parameters
        self.crossover_distance = 0.15  # Lateral distance for crossover motion
        self.step_height = 0.10  # Clearance height during swing
        self.forward_offset = 0.02  # Slight forward component during steps
        
        # Base foot positions (neutral stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters for circular orbit
        self.lateral_velocity = 0.3  # Rightward velocity in body frame (+y)
        self.yaw_rate = 0.8  # Counter-clockwise yaw rotation
        self.forward_velocity = 0.05  # Slight forward component

    def update_base_motion(self, phase, dt):
        """
        Continuous lateral velocity and yaw rate for circular orbit trajectory.
        Base moves rightward (body +y) while rotating counter-clockwise.
        """
        # Constant lateral and yaw velocities throughout the cycle
        vx = self.forward_velocity
        vy = self.lateral_velocity
        vz = 0.0
        
        yaw_rate = self.yaw_rate
        roll_rate = 0.0
        pitch_rate = 0.0
        
        # Velocity commands in world frame
        # Convert body frame velocity to world frame
        vel_body = np.array([vx, vy, vz])
        R = quat_to_rotation_matrix(self.root_quat)
        self.vel_world = R @ vel_body
        
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
        Compute foot trajectory for crossover stepping pattern.
        
        Phase structure:
        - [0.0, 0.25]: Outer legs (FL, RL) swing inward (cross)
        - [0.25, 0.5]: Inner legs (FR, RR) swing outward (uncross)
        - [0.5, 0.75]: Inner legs (FR, RR) swing outward (cross over outer)
        - [0.75, 1.0]: Outer legs (FL, RL) swing outward (uncross)
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is outer (FL, RL) or inner (FR, RR)
        is_outer = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        foot = base_pos.copy()
        
        if is_outer:
            # Outer legs: FL, RL
            # Swing inward (cross) during [0.0, 0.25]
            # Stance during [0.25, 0.75]
            # Swing outward (uncross) during [0.75, 1.0]
            
            if phase < 0.25:
                # Swing phase: cross inward (move rightward, toward +y)
                progress = phase / 0.25
                foot = self._swing_trajectory(
                    base_pos,
                    base_pos + np.array([self.forward_offset if is_front else -self.forward_offset, 
                                        self.crossover_distance, 0.0]),
                    progress
                )
                
            elif phase < 0.75:
                # Stance phase: maintain crossed position
                foot[0] += self.forward_offset if is_front else -self.forward_offset
                foot[1] += self.crossover_distance
                
            else:
                # Swing phase: uncross outward (move leftward, toward -y)
                progress = (phase - 0.75) / 0.25
                start_pos = base_pos + np.array([self.forward_offset if is_front else -self.forward_offset,
                                                 self.crossover_distance, 0.0])
                foot = self._swing_trajectory(start_pos, base_pos, progress)
                
        else:
            # Inner legs: FR, RR
            # Stance during [0.0, 0.25]
            # Swing outward (uncross) during [0.25, 0.5]
            # Swing outward (cross over) during [0.5, 0.75]
            # Stance during [0.75, 1.0]
            
            if phase < 0.25:
                # Stance phase: maintain inner position
                foot = base_pos.copy()
                
            elif phase < 0.5:
                # Swing phase: uncross outward (move leftward, toward -y)
                progress = (phase - 0.25) / 0.25
                end_pos = base_pos + np.array([self.forward_offset if is_front else -self.forward_offset,
                                              -self.crossover_distance * 0.5, 0.0])
                foot = self._swing_trajectory(base_pos, end_pos, progress)
                
            elif phase < 0.75:
                # Swing phase: cross outward further (move more leftward)
                progress = (phase - 0.5) / 0.25
                start_pos = base_pos + np.array([self.forward_offset if is_front else -self.forward_offset,
                                                -self.crossover_distance * 0.5, 0.0])
                end_pos = base_pos + np.array([self.forward_offset if is_front else -self.forward_offset,
                                              -self.crossover_distance, 0.0])
                foot = self._swing_trajectory(start_pos, end_pos, progress)
                
            else:
                # Stance phase: maintain crossed outer position
                foot[0] += self.forward_offset if is_front else -self.forward_offset
                foot[1] -= self.crossover_distance
        
        return foot

    def _swing_trajectory(self, start_pos, end_pos, progress):
        """
        Generate swing trajectory with arc clearance.
        
        Args:
            start_pos: Starting foot position [x, y, z]
            end_pos: Ending foot position [x, y, z]
            progress: Phase progress in [0, 1]
        
        Returns:
            Current foot position along swing trajectory
        """
        # Interpolate x and y linearly
        foot = start_pos + (end_pos - start_pos) * progress
        
        # Add vertical arc for clearance
        arc_height = self.step_height * np.sin(np.pi * progress)
        foot[2] += arc_height
        
        return foot