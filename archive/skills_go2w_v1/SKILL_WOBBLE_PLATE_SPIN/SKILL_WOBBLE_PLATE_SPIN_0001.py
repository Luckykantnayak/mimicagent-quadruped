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
        
        # Compute pitch and roll rates based on quadrant
        # Use smooth sinusoidal profiles for wobble
        if phase < 0.25:
            # Quadrant 1: forward-pitch, right-roll
            local_phase = phase / 0.25
            pitch_rate = self.pitch_amp * 2 * np.pi * self.freq * np.cos(2 * np.pi * local_phase)
            roll_rate = self.roll_amp * 2 * np.pi * self.freq * np.cos(2 * np.pi * local_phase)
        elif phase < 0.5:
            # Quadrant 2: backward-pitch, left-roll
            local_phase = (phase - 0.25) / 0.25
            pitch_rate = -self.pitch_amp * 2 * np.pi * self.freq * np.cos(2 * np.pi * local_phase)
            roll_rate = -self.roll_amp * 2 * np.pi * self.freq * np.cos(2 * np.pi * local_phase)
        elif phase < 0.75:
            # Quadrant 3: forward-pitch, left-roll
            local_phase = (phase - 0.5) / 0.25
            pitch_rate = self.pitch_amp * 2 * np.pi * self.freq * np.cos(2 * np.pi * local_phase)
            roll_rate = -self.roll_amp * 2 * np.pi * self.freq * np.cos(2 * np.pi * local_phase)
        else:
            # Quadrant 4: backward-pitch, right-roll
            local_phase = (phase - 0.75) / 0.25
            pitch_rate = -self.pitch_amp * 2 * np.pi * self.freq * np.cos(2 * np.pi * local_phase)
            roll_rate = self.roll_amp * 2 * np.pi * self.freq * np.cos(2 * np.pi * local_phase)
        
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
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg type
        is_front = leg_name.startswith('F')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compute pitch and roll adjustment factors
        # These shift feet to maintain contact as base tilts
        if phase < 0.25:
            # Forward-pitch, right-roll
            local_phase = phase / 0.25
            pitch_factor = np.sin(np.pi * local_phase)  # forward
            roll_factor = np.sin(np.pi * local_phase)   # right
        elif phase < 0.5:
            # Backward-pitch, left-roll
            local_phase = (phase - 0.25) / 0.25
            pitch_factor = -np.sin(np.pi * local_phase)  # backward
            roll_factor = -np.sin(np.pi * local_phase)   # left
        elif phase < 0.75:
            # Forward-pitch, left-roll
            local_phase = (phase - 0.5) / 0.25
            pitch_factor = np.sin(np.pi * local_phase)   # forward
            roll_factor = -np.sin(np.pi * local_phase)   # left
        else:
            # Backward-pitch, right-roll
            local_phase = (phase - 0.75) / 0.25
            pitch_factor = -np.sin(np.pi * local_phase)  # backward
            roll_factor = np.sin(np.pi * local_phase)    # right
        
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
        # This creates minor repositioning as body frame rotates relative to world
        yaw_phase = phase * 2 * np.pi
        if leg_name.startswith('FL'):
            foot[0] += 0.01 * np.sin(yaw_phase)
            foot[1] += 0.01 * np.cos(yaw_phase)
        elif leg_name.startswith('FR'):
            foot[0] += 0.01 * np.sin(yaw_phase + np.pi/2)
            foot[1] += 0.01 * np.cos(yaw_phase + np.pi/2)
        elif leg_name.startswith('RL'):
            foot[0] += 0.01 * np.sin(yaw_phase + np.pi)
            foot[1] += 0.01 * np.cos(yaw_phase + np.pi)
        elif leg_name.startswith('RR'):
            foot[0] += 0.01 * np.sin(yaw_phase + 3*np.pi/2)
            foot[1] += 0.01 * np.cos(yaw_phase + 3*np.pi/2)
        
        return foot