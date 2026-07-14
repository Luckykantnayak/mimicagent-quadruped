from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_WOBBLE_PLATE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Wobble Plate Spin: Continuous in-place yaw rotation (360 degrees per cycle)
    with superimposed wobbling motion created by coupled pitch and roll oscillations.
    
    - Base executes constant yaw rotation (360 deg/cycle)
    - Pitch and roll oscillate sinusoidally with phase offsets
    - All four feet remain in continuous ground contact
    - Feet adjust positions in body frame to maintain ground contact during wobble-spin
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Yaw rotation parameters
        self.yaw_rate_constant = 2.0 * np.pi  # 360 degrees per cycle (in phase time)
        
        # Wobble parameters (pitch and roll oscillations)
        self.wobble_amplitude_pitch = 0.15  # radians (~8.6 degrees)
        self.wobble_amplitude_roll = 0.15   # radians (~8.6 degrees)
        self.wobble_freq = 2.0  # 2 complete wobble cycles per phase cycle (4 quadrants)
        
        # Foot adjustment scaling for maintaining ground contact
        self.foot_adjust_scale = 0.3  # scaling factor for foot displacement compensation
        
    def update_base_motion(self, phase, dt):
        """
        Update base motion with constant yaw rotation and coupled pitch/roll wobble.
        
        Yaw: constant positive rate (counter-clockwise)
        Pitch: sinusoidal oscillation with 2 cycles per phase
        Roll: sinusoidal oscillation with 2 cycles per phase, phase-shifted 90 degrees
        
        The phase shift creates the quadrant pattern:
        - [0.0-0.25]: pitch forward, roll right
        - [0.25-0.5]: pitch backward, roll left
        - [0.5-0.75]: pitch forward, roll left
        - [0.75-1.0]: pitch backward, roll right
        """
        
        # Constant yaw rate
        yaw_rate = self.yaw_rate_constant
        
        # Wobble rates (sinusoidal pitch and roll)
        # Pitch: positive = nose down, negative = nose up
        # Roll: positive = right down, negative = left down
        wobble_phase = 2.0 * np.pi * self.wobble_freq * phase
        
        # Pitch rate: derivative of pitch angle
        # pitch(t) = A_pitch * sin(wobble_phase)
        # pitch_rate(t) = A_pitch * wobble_freq * 2π * cos(wobble_phase)
        pitch_rate = self.wobble_amplitude_pitch * self.wobble_freq * 2.0 * np.pi * np.cos(wobble_phase)
        
        # Roll rate: 90-degree phase offset to create quadrant pattern
        # roll(t) = A_roll * sin(wobble_phase + π/2) = A_roll * cos(wobble_phase)
        # roll_rate(t) = -A_roll * wobble_freq * 2π * sin(wobble_phase)
        roll_rate = -self.wobble_amplitude_roll * self.wobble_freq * 2.0 * np.pi * np.sin(wobble_phase)
        
        # Set velocities (world frame)
        self.vel_world = np.array([0.0, 0.0, 0.0])  # No translation
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
        Compute foot position in body frame to maintain ground contact while base wobbles and spins.
        
        Strategy:
        - Start from base foot position
        - Apply adjustments to compensate for pitch and roll angles
        - Feet extend in opposite direction to base tilt to maintain ground contact
        
        Pitch compensation: forward pitch -> rear feet extend back, front feet extend forward
        Roll compensation: right roll -> left feet extend left, right feet extend right
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute current wobble angles (integrated from rates)
        wobble_phase = 2.0 * np.pi * self.wobble_freq * phase
        pitch_angle = self.wobble_amplitude_pitch * np.sin(wobble_phase)
        roll_angle = self.wobble_amplitude_roll * np.cos(wobble_phase)
        
        # Determine leg position (front/rear, left/right)
        is_front = leg_name.startswith('F')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Pitch compensation (x-direction in body frame)
        # Forward pitch (positive): front legs extend forward, rear legs extend back
        # Backward pitch (negative): opposite
        if is_front:
            foot[0] += self.foot_adjust_scale * pitch_angle
        else:
            foot[0] -= self.foot_adjust_scale * pitch_angle
        
        # Roll compensation (y-direction in body frame)
        # Right roll (positive): left legs extend left, right legs extend right
        # Left roll (negative): opposite
        if is_left:
            foot[1] -= self.foot_adjust_scale * roll_angle
        else:
            foot[1] += self.foot_adjust_scale * roll_angle
        
        # Z compensation: feet lower slightly to maintain contact during tilt
        # The tilting base causes geometric height variation; compensate to keep feet on ground
        # Approximate effect: z adjustment proportional to tilt magnitude
        tilt_magnitude = np.sqrt(pitch_angle**2 + roll_angle**2)
        foot[2] -= 0.5 * self.foot_adjust_scale * tilt_magnitude
        
        return foot