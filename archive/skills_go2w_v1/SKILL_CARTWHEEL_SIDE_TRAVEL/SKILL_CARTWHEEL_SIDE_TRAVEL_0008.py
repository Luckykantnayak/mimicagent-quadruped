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
    - Legs trace circular arcs synchronized with roll rotation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for dynamic cartwheel (2 seconds per cycle)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Cartwheel parameters
        self.roll_rate_max = 2.0 * np.pi * self.freq  # 360 degrees per cycle
        self.lateral_velocity = 0.2  # Reduced rightward velocity (m/s)
        self.forward_velocity = 0.05  # Slight forward momentum
        
        # Reduced leg arc parameters
        self.arc_radius = 0.22  # Reduced radius for better reachability
        self.arc_height_offset = 0.08  # Reduced vertical offset
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity tracking
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Track accumulated roll angle for trajectory compensation
        self.accumulated_roll = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base motion with continuous roll rotation and lateral displacement.
        Enhanced with higher vertical motion during inversion phase.
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
        
        # Track accumulated roll angle for compensation
        self.accumulated_roll += roll_rate * dt
        
        # Lateral velocity profile: sustained through cartwheel
        if phase < 0.1:
            vy = self.lateral_velocity * (phase / 0.1)
        elif phase < 0.85:
            vy = self.lateral_velocity
        else:
            vy = self.lateral_velocity * (1.0 - (phase - 0.85) / 0.15)
        
        # Forward velocity: slight momentum throughout
        vx = self.forward_velocity
        
        # Enhanced vertical velocity: larger amplitude during inversion to create workspace
        # Peak lift during inversion phase (0.3-0.6)
        if phase < 0.5:
            vz = 0.7 * np.sin(2 * np.pi * phase)
        else:
            vz = 0.5 * np.sin(2 * np.pi * phase)
        
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
        Compute foot position in body frame using roll-angle-compensated trajectories.
        Reduced arc amplitudes and smoother transitions to avoid joint limit violations.
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if right or left leg
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Compute current roll angle from phase (approximate)
        roll_angle = 2.0 * np.pi * phase
        
        if is_right_leg:
            # Right legs: FR, RR
            if phase < 0.25:
                # Stance phase: planted as pivot with slight compression
                foot = base_pos.copy()
                foot[2] += 0.01 * np.sin(np.pi * phase / 0.25)
                return foot
            else:
                # Swing phase: reduced arc overhead from 0.25 to 1.0
                swing_progress = (phase - 0.25) / 0.75
                # Reduced to 180 degree arc instead of 270
                arc_angle = swing_progress * np.pi
                
                # Reduced amplitude circular arc
                foot = base_pos.copy()
                foot[1] += self.arc_radius * np.sin(arc_angle) * 0.25
                foot[2] += self.arc_radius * (1.0 - np.cos(arc_angle))
                
                # Compensation for inverted phases: when roll > 90 deg, adjust Z
                if roll_angle > 0.5 * np.pi and roll_angle < 1.5 * np.pi:
                    # During inversion, reduce Z extension to maintain reachability
                    inversion_factor = np.sin(roll_angle)
                    foot[2] *= (0.6 + 0.4 * abs(inversion_factor))
                
                return foot
        else:
            # Left legs: FL, RL
            if phase < 0.25:
                # Early swing: arc upward as body tilts right
                swing_progress = phase / 0.25
                arc_angle = swing_progress * 0.5 * np.pi  # 90 degrees
                
                foot = base_pos.copy()
                foot[1] -= self.arc_radius * np.sin(arc_angle) * 0.25
                foot[2] += self.arc_radius * (1.0 - np.cos(arc_angle)) + self.arc_height_offset
                
                return foot
            elif phase < 0.5:
                # Continue swing through inversion: overhead to contact
                swing_progress = (phase - 0.25) / 0.25
                arc_angle = 0.5 * np.pi + swing_progress * 0.5 * np.pi  # 90 to 180 degrees
                
                foot = base_pos.copy()
                foot[1] -= self.arc_radius * np.sin(arc_angle) * 0.25
                # Smooth descent to contact position
                z_height = self.arc_radius * (1.0 - np.cos(arc_angle)) + self.arc_height_offset
                # Reduce height as approaching contact
                z_height *= (1.0 - swing_progress * 0.6)
                foot[2] += z_height
                
                return foot
            elif phase < 0.75:
                # Stance phase during inversion: must compensate for base roll
                # When inverted (roll ~180-270 deg), foot needs to reach ground
                foot = base_pos.copy()
                
                # During inverted stance, reduce Z (or make negative) to reach ground
                stance_progress = (phase - 0.5) / 0.25
                if roll_angle > np.pi:
                    # Inverted: foot must extend downward in body frame (negative Z in rotated body)
                    foot[2] = base_pos[2] * 0.5 - 0.05  # Reduced extension
                else:
                    foot[2] += 0.01
                
                return foot
            else:
                # Return to nominal: smooth transition to base stance
                return_progress = (phase - 0.75) / 0.25
                foot = base_pos.copy()
                
                # Smooth blend to nominal
                if roll_angle > 1.5 * np.pi:
                    # Transitioning from inverted to upright
                    z_offset = 0.02 * (1.0 - return_progress)
                    foot[2] += z_offset
                else:
                    foot[2] += 0.01 * (1.0 - return_progress)
                
                return foot