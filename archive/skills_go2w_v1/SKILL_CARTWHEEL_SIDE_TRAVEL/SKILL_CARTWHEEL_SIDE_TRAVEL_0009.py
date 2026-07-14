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
    - Legs trace constrained circular arcs synchronized with roll rotation
    - Base height maintained within safe envelope during inversion
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for dynamic cartwheel (2 seconds per cycle)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Cartwheel parameters
        self.roll_rate_max = 2.0 * np.pi * self.freq  # 360 degrees per cycle
        self.lateral_velocity = 0.2  # Controlled rightward velocity (m/s)
        self.forward_velocity = 0.05  # Slight forward momentum
        
        # Reduced leg arc parameters to maintain reachability
        self.arc_radius = 0.18  # Conservative radius for safe knee angles
        self.arc_height_offset = 0.03  # Minimal vertical offset
        
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
        Update base motion with continuous roll rotation and constrained vertical displacement.
        Vertical motion limited to keep base height within 0.1-0.68m envelope.
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
        
        # Constrained vertical velocity to maintain base height < 0.68m
        # Target peak height: 0.55m, nominal: 0.30m, rise needed: 0.25m
        # Reduced amplitude with asymmetric profile
        if phase < 0.25:
            # Ascent during initial tilt
            vz = 0.35 * np.sin(2 * np.pi * phase)
        elif phase < 0.5:
            # Peak and early descent during inversion
            vz = 0.35 * np.sin(2 * np.pi * phase)
        elif phase < 0.85:
            # Continued descent during recovery
            vz = 0.3 * np.sin(2 * np.pi * phase)
        else:
            # Final damping to ensure return to nominal height
            damping_progress = (phase - 0.85) / 0.15
            vz = 0.3 * np.sin(2 * np.pi * phase) * (1.0 - damping_progress)
        
        # Additional soft limiting based on current height
        current_height = self.root_pos[2]
        if current_height > 0.58:
            # Attenuate upward velocity near ceiling
            vz = min(vz, -0.2)  # Force downward
        elif current_height < 0.25:
            # Prevent sinking too low
            vz = max(vz, 0.1)
        
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
        Compute foot position in body frame using constrained arc trajectories.
        Arc amplitudes reduced to maintain knee joint angles within safe limits.
        Inverted stance phases use ground-aware positioning.
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if right or left leg
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Compute current roll angle from phase (approximate)
        roll_angle = 2.0 * np.pi * phase
        
        # Adaptive arc scaling based on phase (reduce during high base height phases)
        if phase > 0.2 and phase < 0.6:
            # Inversion phase: reduce arc amplitude
            arc_scale = 0.75
        else:
            arc_scale = 1.0
        
        effective_radius = self.arc_radius * arc_scale
        effective_height_offset = self.arc_height_offset * arc_scale
        
        if is_right_leg:
            # Right legs: FR, RR
            if phase < 0.25:
                # Stance phase: planted as pivot with minimal motion
                foot = base_pos.copy()
                # Slight compression as body rolls
                foot[2] += 0.01 * np.sin(np.pi * phase / 0.25)
                return foot
            elif phase < 0.75:
                # Swing phase: constrained arc overhead - reduced to 120 degrees
                swing_progress = (phase - 0.25) / 0.5
                # Limit arc to 120 degrees (2π/3) instead of 180
                arc_angle = swing_progress * (2.0 * np.pi / 3.0)
                
                # Constrained amplitude circular arc
                foot = base_pos.copy()
                foot[1] += effective_radius * np.sin(arc_angle) * 0.2
                foot[2] += effective_radius * (1.0 - np.cos(arc_angle))
                
                # Additional roll compensation during inversion
                if roll_angle > 0.5 * np.pi and roll_angle < 1.5 * np.pi:
                    # Reduce Z extension during inverted phases
                    foot[2] *= 0.7
                
                return foot
            else:
                # Return phase: explicit descent to ground
                return_progress = (phase - 0.75) / 0.25
                foot = base_pos.copy()
                
                # Smooth vertical descent from swing arc to ground
                if return_progress < 0.6:
                    # First 60%: descend from arc
                    descent_height = effective_radius * 0.7 * (1.0 - return_progress / 0.6)
                    foot[2] += descent_height
                else:
                    # Final 40%: settle to nominal with minimal offset
                    settle_progress = (return_progress - 0.6) / 0.4
                    foot[2] += 0.01 * (1.0 - settle_progress)
                
                return foot
        else:
            # Left legs: FL, RL
            if phase < 0.25:
                # Early swing: constrained arc upward
                swing_progress = phase / 0.25
                arc_angle = swing_progress * 0.4 * np.pi  # Reduced to 72 degrees
                
                foot = base_pos.copy()
                foot[1] -= effective_radius * np.sin(arc_angle) * 0.2
                foot[2] += effective_radius * (1.0 - np.cos(arc_angle)) + effective_height_offset
                
                return foot
            elif phase < 0.5:
                # Continue swing through inversion: overhead to pre-contact
                swing_progress = (phase - 0.25) / 0.25
                arc_angle = 0.4 * np.pi + swing_progress * 0.4 * np.pi  # 72 to 144 degrees
                
                foot = base_pos.copy()
                foot[1] -= effective_radius * np.sin(arc_angle) * 0.2
                # Smooth descent toward contact position
                z_height = effective_radius * (1.0 - np.cos(arc_angle)) + effective_height_offset
                # Progressively reduce height as approaching contact
                z_height *= (1.0 - swing_progress * 0.5)
                foot[2] += z_height
                
                return foot
            elif phase < 0.75:
                # Stance phase during inversion: ground-aware positioning
                stance_progress = (phase - 0.5) / 0.25
                foot = base_pos.copy()
                
                # During inverted stance, foot must maintain ground contact
                # Use conservative Z positioning that accounts for base height
                if roll_angle > np.pi and roll_angle < 1.8 * np.pi:
                    # Inverted: reduce Z to maintain reachability
                    # Target position slightly below nominal to ensure contact
                    foot[2] = base_pos[2] * 0.4 - 0.02
                else:
                    # Transitioning toward upright
                    foot[2] = base_pos[2] * 0.6
                
                return foot
            else:
                # Return to nominal: smooth transition to base stance
                return_progress = (phase - 0.75) / 0.25
                foot = base_pos.copy()
                
                # Smooth blend to nominal stance position
                z_offset = 0.02 * (1.0 - return_progress)
                foot[2] += z_offset
                
                return foot