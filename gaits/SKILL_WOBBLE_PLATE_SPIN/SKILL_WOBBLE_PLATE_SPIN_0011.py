from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_WOBBLE_PLATE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Wobble-plate-spin gait: continuous in-place yaw rotation (360° per cycle)
    combined with coordinated pitch-roll wobbling motion.
    
    Motion pattern over one cycle:
    - [0.0, 0.25]: forward pitch + right roll + yaw CW 90°
    - [0.25, 0.5]: backward pitch + left roll + yaw CW 90°
    - [0.5, 0.75]: forward pitch + left roll + yaw CW 90°
    - [0.75, 1.0]: backward pitch + right roll + yaw CW 90°
    
    All four legs maintain continuous ground contact throughout.
    Legs adjust stance positions in body frame to track world-fixed
    ground contact points as base rotates and wobbles.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # 0.5 Hz = 2 seconds per full cycle
        
        # Base foot positions in body frame - these will be our fixed world positions
        # transformed to initial body frame
        self.initial_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compute initial world positions (at phase 0, base is at identity orientation)
        self.feet_world_positions = {}
        for leg_name in leg_names:
            self.feet_world_positions[leg_name] = self.initial_feet_pos_body[leg_name].copy()
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters - reduced amplitudes for safety
        self.yaw_rate = 2.0 * np.pi * self.freq  # 360° per cycle
        self.max_pitch_angle = 0.12  # Peak tilt angle in radians (~7 degrees)
        self.max_roll_angle = 0.12   # Peak tilt angle in radians (~7 degrees)
        
        # Base height for geometric compensation
        self.base_height = 0.28  # Approximate base height above ground

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-derived angles for smooth periodic motion.
        """
        
        # Compute target angles directly from phase (no integration drift)
        target_yaw = 2.0 * np.pi * phase  # 0 to 360° over full cycle
        
        # Pitch oscillates: forward [0, 0.25], back [0.25, 0.5], forward [0.5, 0.75], back [0.75, 1.0]
        # Use sin(4π*phase) which completes 2 full oscillations per cycle
        target_pitch = self.max_pitch_angle * np.sin(4.0 * np.pi * phase)
        
        # Roll oscillates with 90° phase offset: right [0, 0.25], left [0.25, 0.75], right [0.75, 1.0]
        # Use cos(4π*phase)
        target_roll = self.max_roll_angle * np.cos(4.0 * np.pi * phase)
        
        # Convert target angles to quaternion
        # Order: roll (X), pitch (Y), yaw (Z)
        self.root_quat = quat_from_euler_xyz(np.array([target_roll, target_pitch, target_yaw]))
        
        # Base stays at origin for in-place motion
        self.root_pos = np.array([0.0, 0.0, 0.0])
        
        # Set velocity outputs for visualization consistency
        # Compute angular velocities as derivatives of target angles
        yaw_rate = self.yaw_rate
        pitch_rate = self.max_pitch_angle * 4.0 * np.pi * np.cos(4.0 * np.pi * phase)
        roll_rate = -self.max_roll_angle * 4.0 * np.pi * np.sin(4.0 * np.pi * phase)
        
        self.vel_world = np.array([0.0, 0.0, 0.0])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame to maintain fixed world-space ground contact.
        
        Strategy:
        1. Feet remain at fixed world positions throughout motion
        2. Transform world position to current body frame using inverse of base rotation
        3. Apply geometric corrections for pitch/roll to keep feet on ground plane
        """
        
        # Get fixed world position for this foot
        foot_world = self.feet_world_positions[leg_name].copy()
        
        # Compute current base orientation angles from phase
        current_yaw = 2.0 * np.pi * phase
        current_pitch = self.max_pitch_angle * np.sin(4.0 * np.pi * phase)
        current_roll = self.max_roll_angle * np.cos(4.0 * np.pi * phase)
        
        # Determine leg characteristics
        is_front = leg_name.startswith('F')
        is_right = leg_name.endswith('R')
        
        # Transform world position to body frame
        # Apply inverse yaw rotation (rotate by -yaw)
        cos_yaw = np.cos(-current_yaw)
        sin_yaw = np.sin(-current_yaw)
        
        x_body = cos_yaw * foot_world[0] - sin_yaw * foot_world[1]
        y_body = sin_yaw * foot_world[0] + cos_yaw * foot_world[1]
        z_body = foot_world[2]
        
        # Apply geometric compensation for pitch
        # When base pitches forward (+pitch), front of base moves down
        # For front legs: foot appears to move backward (-x) and up (+z) in body frame
        # For rear legs: foot appears to move forward (+x) and down (-z) in body frame
        if is_front:
            x_body -= self.base_height * np.tan(current_pitch)
            z_body += self.base_height * (1.0 - np.cos(current_pitch))
        else:
            x_body += self.base_height * np.tan(current_pitch)
            z_body += self.base_height * (1.0 - np.cos(current_pitch))
        
        # Apply geometric compensation for roll
        # When base rolls right (+roll), right side of base moves down
        # For right legs: foot appears to move left (-y) and up (+z) in body frame
        # For left legs: foot appears to move right (+y) and down (-z) in body frame
        if is_right:
            y_body -= self.base_height * np.tan(current_roll)
            z_body += self.base_height * (1.0 - np.cos(current_roll))
        else:
            y_body += self.base_height * np.tan(current_roll)
            z_body += self.base_height * (1.0 - np.cos(current_roll))
        
        # Clamp z to safe range to prevent extreme leg extensions
        z_body = np.clip(z_body, -0.4, -0.15)
        
        return np.array([x_body, y_body, z_body])