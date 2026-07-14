from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_WOBBLE_PLATE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Wobble Plate Spin: Continuous 360° yaw rotation with synchronized pitch-roll wobbling.
    
    - Base rotates continuously in yaw (constant angular velocity)
    - Pitch and roll oscillate in coordinated quadrant pattern
    - All four feet maintain continuous ground contact
    - Feet adjust dynamically in body frame to track wobbling base
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # 2 seconds per complete cycle
        
        # Base foot positions (body frame reference)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Wobble parameters
        self.pitch_amp = np.deg2rad(12)  # ±12° pitch amplitude
        self.roll_amp = np.deg2rad(12)   # ±12° roll amplitude
        self.yaw_rate = 2 * np.pi * self.freq  # 360° per cycle (constant)
        
        # Foot adjustment amplitude (to maintain contact during wobble)
        self.foot_adjust_x = 0.03  # forward/backward adjustment (reduced for workspace)
        self.foot_adjust_y = 0.02  # lateral adjustment (reduced for workspace)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Track integrated pitch and roll angles for Z-compensation
        self.current_pitch = 0.0
        self.current_roll = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base angular velocities: constant yaw + wobbling pitch/roll.
        
        Phase structure:
        [0.0, 0.25]: forward-pitch + right-roll
        [0.25, 0.5]: backward-pitch + left-roll
        [0.5, 0.75]: forward-pitch + left-roll
        [0.75, 1.0]: backward-pitch + right-roll
        """
        
        # Zero linear velocity (in-place motion)
        vx, vy, vz = 0.0, 0.0, 0.0
        
        # Constant yaw rate
        yaw_rate = self.yaw_rate
        
        # Compute target pitch and roll angles based on phase using smooth sinusoids
        angle_phase = phase * 2 * np.pi
        
        # Pitch: oscillates with period 0.5 (two full cycles per motion cycle)
        # Quadrant pattern: forward-back-forward-back
        pitch_target = self.pitch_amp * np.sin(4 * np.pi * phase)
        
        # Roll: oscillates with quadrant-specific phase shifts
        # Pattern: right-left-left-right
        if phase < 0.5:
            roll_target = self.roll_amp * np.sin(2 * np.pi * phase)
        else:
            roll_target = -self.roll_amp * np.sin(2 * np.pi * (phase - 0.5))
        
        # Compute rates as derivatives of target angles
        pitch_rate = (pitch_target - self.current_pitch) / max(dt, 1e-6)
        roll_rate = (roll_target - self.current_roll) / max(dt, 1e-6)
        
        # Update tracked angles
        self.current_pitch = pitch_target
        self.current_roll = roll_target
        
        # Set world-frame commands
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame with adjustments for wobble.
        
        Feet remain in contact (no liftoff), but shift position in body frame
        to accommodate pitch/roll tilts and yaw rotation tracking.
        Includes Z-axis compensation to maintain ground contact.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Store original reference position for Z-compensation calculation
        original_foot_x = self.base_feet_pos_body[leg_name][0]
        original_foot_y = self.base_feet_pos_body[leg_name][1]
        
        # Determine leg type
        is_front = leg_name.startswith('F')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Smooth continuous adjustment factors
        # Pitch factor: forward-back-forward-back pattern
        pitch_factor = np.sin(4 * np.pi * phase)
        
        # Roll factor: right-left-left-right pattern
        if phase < 0.5:
            roll_factor = np.sin(2 * np.pi * phase)
        else:
            roll_factor = -np.sin(2 * np.pi * (phase - 0.5))
        
        # Scale adjustments based on combined tilt magnitude (reduce when both pitch and roll are large)
        tilt_magnitude = np.sqrt(self.current_pitch**2 + self.current_roll**2)
        max_tilt = np.sqrt(2) * self.pitch_amp  # maximum when both at peak
        adjustment_scale = 1.0 - 0.2 * (tilt_magnitude / max_tilt)  # reduce by up to 20% at max tilt
        
        # Apply forward/backward shift based on pitch (front legs extend forward, rear extend back)
        if is_front:
            foot[0] += self.foot_adjust_x * pitch_factor * adjustment_scale
        else:
            foot[0] -= self.foot_adjust_x * pitch_factor * adjustment_scale
        
        # Apply lateral shift based on roll (left legs shift left, right legs shift right)
        if is_left:
            foot[1] += self.foot_adjust_y * roll_factor * adjustment_scale
        else:
            foot[1] -= self.foot_adjust_y * roll_factor * adjustment_scale
        
        # Add subtle cyclic variation to track yaw rotation smoothly
        yaw_phase = phase * 2 * np.pi
        yaw_adjust = 0.008  # slightly reduced for workspace margin
        if leg_name.startswith('FL'):
            foot[0] += yaw_adjust * np.sin(yaw_phase)
            foot[1] += yaw_adjust * np.cos(yaw_phase)
        elif leg_name.startswith('FR'):
            foot[0] += yaw_adjust * np.sin(yaw_phase + np.pi/2)
            foot[1] += yaw_adjust * np.cos(yaw_phase + np.pi/2)
        elif leg_name.startswith('RL'):
            foot[0] += yaw_adjust * np.sin(yaw_phase + np.pi)
            foot[1] += yaw_adjust * np.cos(yaw_phase + np.pi)
        elif leg_name.startswith('RR'):
            foot[0] += yaw_adjust * np.sin(yaw_phase + 3*np.pi/2)
            foot[1] += yaw_adjust * np.cos(yaw_phase + 3*np.pi/2)
        
        # Z-compensation to maintain ground contact during pitch/roll
        # CORRECTED LOGIC:
        # When base pitches forward (positive pitch), front feet (positive X) move DOWN in world frame
        # To maintain ground contact, they need to be commanded HIGHER in body frame (positive Z)
        # Similarly for roll: when rolling right (positive roll), right feet (negative Y) move DOWN in world
        # They need positive Z adjustment in body frame
        
        # Use ORIGINAL foot positions for geometric calculation (not adjusted positions)
        # Pitch compensation: positive pitch requires positive Z for positive X (front feet)
        # For small angles: Z adjustment ≈ X * pitch_angle
        pitch_z_adjust = original_foot_x * self.current_pitch
        
        # Roll compensation: positive roll (right side down) requires positive Z for negative Y (right feet)
        # For small angles: Z adjustment ≈ -Y * roll_angle (negative because right is negative Y)
        roll_z_adjust = -original_foot_y * self.current_roll
        
        # Apply Z-compensation (sum of pitch and roll contributions)
        foot[2] += pitch_z_adjust + roll_z_adjust
        
        return foot