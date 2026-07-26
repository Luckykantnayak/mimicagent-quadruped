from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_DIAGONAL_ROLL_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal roll drift skill: robot travels diagonally forward-right with continuous
    roll oscillations and perpendicular lateral drift, creating a serpentine path.
    
    - All four feet maintain ground contact throughout (shuffling gait)
    - Roll oscillates sinusoidally (left-right-left-right per cycle)
    - Lateral velocity oscillates perpendicular to forward travel
    - Forward velocity is constant, creating net diagonal motion
    - Foot positions in body frame adjust to compensate for roll and drift
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower cycle for smooth roll oscillations
        
        # Base foot positions (BODY frame) - raise z-height to prevent ground penetration
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            foot = v.copy()
            foot[2] += 0.028  # Uniform upward offset for ground clearance margin
            self.base_feet_pos_body[k] = foot
        
        # Time tracking
        self.t = 0.0
        
        # Base state (WORLD frame)
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.vx_forward = 0.4  # Constant forward velocity (m/s)
        self.vy_drift_amp = 0.12  # Lateral drift amplitude (m/s)
        self.roll_rate_amp = 0.6  # Roll rate amplitude (rad/s) - produces ~15-20 deg peak roll
        
        # Foot adjustment parameters for maintaining contact during roll/drift
        self.lateral_adjust_amp = 0.03  # Lateral adjustment amplitude in body frame (m)
        self.forward_slip_rate = 0.08  # Rearward slip per cycle to compensate forward motion (m)
        
        # Vertical compensation parameter to account for roll-induced height changes
        self.z_compensation_amp = 0.022  # Vertical adjustment amplitude (m)

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity, oscillating lateral drift,
        and oscillating roll rate.
        
        - vx: constant forward
        - vy: sinusoidal lateral drift (2 cycles per phase cycle for symmetric drift)
        - roll_rate: sinusoidal roll oscillation (2 cycles per phase cycle)
        """
        # Constant forward velocity
        vx = self.vx_forward
        
        # Lateral drift: oscillates left-right-left-right over one phase cycle
        # Peak left at phase 0.125 and 0.625, peak right at 0.375 and 0.875
        vy = -self.vy_drift_amp * np.sin(4 * np.pi * phase)
        
        # Roll rate: oscillates to produce left-right-left-right roll over one cycle
        # Negative roll rate in [0, 0.25] and [0.5, 0.75] -> roll left
        # Positive roll rate in [0.25, 0.5] and [0.75, 1.0] -> roll right
        roll_rate = -self.roll_rate_amp * np.cos(4 * np.pi * phase)
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, 0.0])
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
        Compute foot position in body frame with adjustments for:
        1. Lateral compensation for roll (left feet move right when rolling left, etc.)
        2. Lateral compensation for drift velocity
        3. Sinusoidal forward-backward slip to compensate for forward base motion
        4. Vertical (z) compensation to maintain ground contact during roll
        
        All feet remain in contact; adjustments are smooth and continuous.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left or right leg
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Lateral adjustment for roll compensation
        # When rolling left (phase 0-0.25, 0.5-0.75), left feet move toward centerline (positive y),
        # right feet move away (negative y). Vice versa when rolling right.
        # Use sinusoidal profile synchronized with roll: sin(4*pi*phase) gives the roll direction
        roll_compensation = self.lateral_adjust_amp * np.sin(4 * np.pi * phase)
        if is_left:
            foot[1] += roll_compensation  # Left feet adjust rightward when rolling left
        else:
            foot[1] -= roll_compensation  # Right feet adjust leftward when rolling left
        
        # Additional lateral adjustment for drift compensation
        # When drifting left (vy < 0), feet need to shift right in body frame to maintain contact
        # Drift is -sin(4*pi*phase), so compensation is +sin(4*pi*phase)
        drift_compensation = 0.5 * self.lateral_adjust_amp * np.sin(4 * np.pi * phase)
        foot[1] += drift_compensation
        
        # Forward-backward slip in body frame to compensate for forward motion
        # Use sinusoidal profile for symmetric oscillation instead of linear accumulation
        forward_compensation = -self.forward_slip_rate * np.sin(2 * np.pi * phase)
        foot[0] += forward_compensation
        
        # Front legs have slightly larger lateral adjustments due to moment arm from roll center
        # Reduced amplification factor from 1.2 to 1.1 to reduce mismatch
        if is_front:
            lateral_amplification = 1.1
            foot[1] = self.base_feet_pos_body[leg_name][1] + (foot[1] - self.base_feet_pos_body[leg_name][1]) * lateral_amplification
        
        # Vertical (z) compensation to account for roll-induced height changes
        # When rolling left (sin(4*pi*phase) > 0), left side goes down, so left feet need to be raised
        # When rolling right (sin(4*pi*phase) < 0), right side goes down, so right feet need to be raised
        roll_phase_signal = np.sin(4 * np.pi * phase)
        if is_left:
            # Left feet: raise when rolling left (positive roll_phase_signal)
            foot[2] += self.z_compensation_amp * roll_phase_signal
        else:
            # Right feet: raise when rolling right (negative roll_phase_signal)
            foot[2] -= self.z_compensation_amp * roll_phase_signal
        
        return foot