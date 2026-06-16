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
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.yaw_rate = 2.0 * np.pi * self.freq  # 360° per cycle
        self.pitch_amplitude = 0.25  # rad/s peak angular rate
        self.roll_amplitude = 0.25   # rad/s peak angular rate
        
        # Leg adjustment parameters for wobble compensation
        self.leg_extension_scale = 0.03  # meters per radian of tilt
        
        # Track integrated angles for leg compensation
        self.integrated_yaw = 0.0
        self.integrated_pitch = 0.0
        self.integrated_roll = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base using constant yaw rate and phase-dependent pitch/roll rates.
        
        Pitch and roll rates follow sinusoidal patterns with 90° phase offset
        and sign reversals to create the figure-8 wobble pattern.
        """
        
        # Constant positive yaw rate for smooth 360° rotation
        yaw_rate = self.yaw_rate
        
        # Pitch rate: positive [0, 0.25], negative [0.25, 0.5], positive [0.5, 0.75], negative [0.75, 1.0]
        # Sinusoidal with period = 0.5 (two full oscillations per cycle)
        pitch_rate = self.pitch_amplitude * np.sin(4 * np.pi * phase)
        
        # Roll rate: 90° phase shift relative to pitch
        # Positive [0, 0.25], negative [0.25, 0.75], positive [0.75, 1.0]
        # Pattern: right [0, 0.25], left [0.25, 0.75], right [0.75, 1.0]
        roll_rate = self.roll_amplitude * np.cos(4 * np.pi * phase)
        
        # Set velocity commands (all linear velocities zero for in-place motion)
        self.vel_world = np.array([0.0, 0.0, 0.0])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Track integrated angles for leg compensation calculations
        self.integrated_yaw += yaw_rate * dt
        self.integrated_pitch += pitch_rate * dt
        self.integrated_roll += roll_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame to maintain ground contact
        while base rotates and wobbles.
        
        Strategy:
        1. Start from base foot position
        2. Apply rotation compensation for integrated yaw
        3. Apply extension/compression for pitch compensation
        4. Apply lateral adjustment for roll compensation
        """
        
        # Get base position for this leg
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is front/rear and left/right
        is_front = leg_name.startswith('F')
        is_right = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Start with base position
        foot_pos = base_pos.copy()
        
        # Apply yaw rotation compensation
        # As base rotates, foot position in body frame rotates oppositely
        # to maintain world-fixed ground contact
        yaw_angle = -self.integrated_yaw  # Negative to compensate base rotation
        cos_yaw = np.cos(yaw_angle)
        sin_yaw = np.sin(yaw_angle)
        
        x_rot = cos_yaw * base_pos[0] - sin_yaw * base_pos[1]
        y_rot = sin_yaw * base_pos[0] + cos_yaw * base_pos[1]
        
        foot_pos[0] = x_rot
        foot_pos[1] = y_rot
        
        # Apply pitch compensation (z-axis adjustment)
        # Forward pitch: front legs compress (z more negative), rear legs extend (z less negative)
        # Backward pitch: opposite
        pitch_adjustment = self.integrated_pitch * self.leg_extension_scale
        if is_front:
            foot_pos[2] -= pitch_adjustment
        else:
            foot_pos[2] += pitch_adjustment
        
        # Apply roll compensation (z-axis adjustment)
        # Right roll: right legs extend (z less negative), left legs compress (z more negative)
        # Left roll: opposite
        roll_adjustment = self.integrated_roll * self.leg_extension_scale
        if is_right:
            foot_pos[2] -= roll_adjustment
        else:
            foot_pos[2] += roll_adjustment
        
        # Add fine wobble tracking using phase-dependent adjustments
        # to ensure smooth continuous motion
        
        # Additional x-y adjustments to maintain stance polygon during wobble
        # Small corrections to prevent legs from drifting during complex motion
        wobble_x_offset = 0.01 * np.sin(2 * np.pi * phase)
        wobble_y_offset = 0.01 * np.cos(2 * np.pi * phase)
        
        if is_front:
            foot_pos[0] += wobble_x_offset
        else:
            foot_pos[0] -= wobble_x_offset
            
        if is_right:
            foot_pos[1] += wobble_y_offset
        else:
            foot_pos[1] -= wobble_y_offset
        
        return foot_pos