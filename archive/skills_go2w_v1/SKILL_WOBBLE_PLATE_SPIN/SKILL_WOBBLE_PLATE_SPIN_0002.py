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
        self.foot_adjust_x = 0.04  # forward/backward adjustment
        self.foot_adjust_y = 0.03  # lateral adjustment
        
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
        
        # Compute target pitch and roll angles based on phase (smooth continuous)
        # Use sinusoidal profiles across full cycle for smoothness
        angle_phase = phase * 2 * np.pi
        
        # Quadrant pattern for pitch: forward-back-forward-back
        if phase < 0.25:
            pitch_target = self.pitch_amp * np.sin(2 * np.pi * phase / 0.5)
        elif phase < 0.5:
            pitch_target = -self.pitch_amp * np.sin(2 * np.pi * (phase - 0.25) / 0.5)
        elif phase < 0.75:
            pitch_target = self.pitch_amp * np.sin(2 * np.pi * (phase - 0.5) / 0.5)
        else:
            pitch_target = -self.pitch_amp * np.sin(2 * np.pi * (phase - 0.75) / 0.5)
        
        # Quadrant pattern for roll: right-left-left-right
        if phase < 0.25:
            roll_target = self.roll_amp * np.sin(2 * np.pi * phase / 0.5)
        elif phase < 0.5:
            roll_target = -self.roll_amp * np.sin(2 * np.pi * (phase - 0.25) / 0.5)
        elif phase < 0.75:
            roll_target = -self.roll_amp * np.sin(2 * np.pi * (phase - 0.5) / 0.5)
        else:
            roll_target = self.roll_amp * np.sin(2 * np.pi * (phase - 0.75) / 0.5)
        
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
        
        # Determine leg type
        is_front = leg_name.startswith('F')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Smooth continuous adjustment factors across full cycle
        angle_phase = phase * 2 * np.pi
        
        # Pitch factor (continuous sinusoidal)
        if phase < 0.25:
            pitch_factor = np.sin(2 * np.pi * phase / 0.5)
        elif phase < 0.5:
            pitch_factor = -np.sin(2 * np.pi * (phase - 0.25) / 0.5)
        elif phase < 0.75:
            pitch_factor = np.sin(2 * np.pi * (phase - 0.5) / 0.5)
        else:
            pitch_factor = -np.sin(2 * np.pi * (phase - 0.75) / 0.5)
        
        # Roll factor (continuous sinusoidal)
        if phase < 0.25:
            roll_factor = np.sin(2 * np.pi * phase / 0.5)
        elif phase < 0.5:
            roll_factor = -np.sin(2 * np.pi * (phase - 0.25) / 0.5)
        elif phase < 0.75:
            roll_factor = -np.sin(2 * np.pi * (phase - 0.5) / 0.5)
        else:
            roll_factor = np.sin(2 * np.pi * (phase - 0.75) / 0.5)
        
        # Apply forward/backward shift based on pitch (front legs extend forward, rear extend back)
        if is_front:
            foot[0] += self.foot_adjust_x * pitch_factor
        else:
            foot[0] -= self.foot_adjust_x * pitch_factor
        
        # Apply lateral shift based on roll (left legs shift left, right legs shift right)
        if is_left:
            foot[1] += self.foot_adjust_y * roll_factor
        else:
            foot[1] -= self.foot_adjust_y * roll_factor
        
        # Add subtle cyclic variation to track yaw rotation smoothly
        yaw_phase = phase * 2 * np.pi
        yaw_adjust = 0.01
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
        # When base pitches forward (positive), front feet need to extend down (negative Z)
        # When base rolls right (positive), right feet need to extend down (negative Z)
        
        # Get foot's horizontal distance from base center
        foot_x = foot[0]
        foot_y = foot[1]
        
        # Compute Z-adjustment based on current pitch and roll angles
        # Pitch causes Z-displacement proportional to X-distance: dZ = -x * sin(pitch)
        # Roll causes Z-displacement proportional to Y-distance: dZ = -y * sin(roll)
        pitch_z_adjust = -foot_x * np.sin(self.current_pitch)
        roll_z_adjust = -foot_y * np.sin(self.current_roll)
        
        # Apply Z-compensation (sum of pitch and roll contributions)
        foot[2] += pitch_z_adjust + roll_z_adjust
        
        return foot