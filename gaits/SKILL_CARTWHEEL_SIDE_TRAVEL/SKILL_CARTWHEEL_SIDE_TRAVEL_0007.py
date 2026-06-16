from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_CARTWHEEL_SIDE_TRAVEL_MotionGenerator(BaseMotionGenerator):
    """
    Cartwheel motion with full 360-degree roll rotation and lateral displacement.
    
    The robot performs a dynamic cartwheel by:
    - Rolling continuously around the x-axis (360 degrees per cycle)
    - Moving laterally (positive y direction) 
    - Sequencing leg contacts: right legs plant [0, 0.25], aerial [0.25, 0.5],
      left legs plant [0.5, 0.75], all legs return [0.75, 1.0]
    - Legs trace circular arcs in body frame synchronized with roll rotation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for dynamic cartwheel (2 seconds per cycle)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Cartwheel parameters
        self.roll_rate_max = 2.0 * np.pi * self.freq  # 360 degrees per cycle
        self.lateral_velocity = 0.4  # Rightward velocity (m/s)
        self.forward_velocity = 0.1  # Slight forward momentum
        
        # Leg arc parameters (circular trajectory in body frame)
        self.arc_radius = 0.35  # Radius of circular leg trajectory
        self.arc_height_offset = 0.15  # Vertical offset for swing arc
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity tracking
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with continuous roll rotation and lateral displacement.
        
        Phase-dependent velocity profile:
        - [0.0, 0.25]: Initiate roll and lateral motion
        - [0.25, 0.5]: Continue through inversion with peak roll rate
        - [0.5, 0.75]: Maintain roll through recovery
        - [0.75, 1.0]: Decelerate to complete rotation
        """
        
        # Roll rate profile: ramp up, sustain, ramp down
        if phase < 0.25:
            # Initiation: accelerate roll
            roll_progress = phase / 0.25
            roll_rate = self.roll_rate_max * np.sin(np.pi * roll_progress / 2)
        elif phase < 0.75:
            # Sustained rotation through inversion and recovery
            roll_rate = self.roll_rate_max
        else:
            # Completion: decelerate roll
            roll_progress = (phase - 0.75) / 0.25
            roll_rate = self.roll_rate_max * np.cos(np.pi * roll_progress / 2)
        
        # Lateral velocity profile: sustained through cartwheel
        if phase < 0.1:
            # Ramp up lateral velocity
            vy = self.lateral_velocity * (phase / 0.1)
        elif phase < 0.85:
            # Sustained lateral motion
            vy = self.lateral_velocity
        else:
            # Ramp down
            vy = self.lateral_velocity * (1.0 - (phase - 0.85) / 0.15)
        
        # Forward velocity: slight momentum throughout
        vx = self.forward_velocity
        
        # Vertical velocity: follows sinusoidal arc during inversion
        # Body rises during early roll, peaks at inversion, descends during recovery
        vz = 0.3 * np.sin(2 * np.pi * phase)
        
        # Set world frame velocities
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame using circular arc trajectories
        synchronized with cartwheel rotation.
        
        Right legs (FR, RR): stance [0, 0.25], swing [0.25, 1.0]
        Left legs (FL, RL): swing [0, 0.5], stance [0.5, 0.75], return [0.75, 1.0]
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if right or left leg
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        if is_right_leg:
            # Right legs: FR, RR
            if phase < 0.25:
                # Stance phase: planted as pivot
                # Slight retraction as body rolls
                foot = base_pos.copy()
                foot[2] += 0.02 * (phase / 0.25)  # Minor compression
                return foot
            else:
                # Swing phase: arc overhead from 0.25 to 1.0
                # Map phase [0.25, 1.0] to arc angle [0, 270 degrees]
                swing_progress = (phase - 0.25) / 0.75
                arc_angle = swing_progress * 1.5 * np.pi  # 270 degrees
                
                # Circular arc in Y-Z plane (body frame)
                # Start from ground, arc overhead, return to ground
                foot = base_pos.copy()
                foot[1] += self.arc_radius * np.sin(arc_angle) * 0.3  # Lateral arc component
                foot[2] += self.arc_radius * (1.0 - np.cos(arc_angle))  # Vertical arc
                
                return foot
        else:
            # Left legs: FL, RL
            if phase < 0.25:
                # Early swing: arc upward as body tilts right
                swing_progress = phase / 0.25
                arc_angle = swing_progress * 0.5 * np.pi  # 90 degrees
                
                foot = base_pos.copy()
                foot[1] -= self.arc_radius * np.sin(arc_angle) * 0.3
                foot[2] += self.arc_radius * (1.0 - np.cos(arc_angle)) + self.arc_height_offset
                
                return foot
            elif phase < 0.5:
                # Continue swing through inversion: overhead to contact
                swing_progress = (phase - 0.25) / 0.25
                arc_angle = 0.5 * np.pi + swing_progress * 0.5 * np.pi  # 90 to 180 degrees
                
                foot = base_pos.copy()
                foot[1] -= self.arc_radius * np.sin(arc_angle) * 0.3
                foot[2] += self.arc_radius * (1.0 - np.cos(arc_angle)) + self.arc_height_offset
                
                return foot
            elif phase < 0.75:
                # Stance phase: planted as pivot during inverted support
                foot = base_pos.copy()
                foot[2] += 0.02 * ((0.75 - phase) / 0.25)  # Minor compression
                return foot
            else:
                # Return to nominal: smooth transition to base stance
                return_progress = (phase - 0.75) / 0.25
                foot = base_pos.copy()
                
                # Smoothly lower from slight elevation to ground
                foot[2] += 0.02 * (1.0 - return_progress)
                
                return foot