from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ORBIT_TILT_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Circular walking pattern with rhythmic base roll oscillation.
    
    - Base moves in a circular orbit via constant forward velocity + constant yaw rate
    - Roll oscillates sinusoidally twice per orbit (left-neutral-right-neutral)
    - Diagonal gait coordination with asymmetric leg extension to support roll tilt
    - Outer legs (relative to roll direction) extend, inner legs retract
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.4  # Full orbit cycle frequency
        
        # Gait parameters
        self.duty = 0.875  # 87.5% stance, 12.5% swing per leg
        self.step_length = 0.12  # Forward step length
        self.step_height = 0.06  # Swing clearance height
        
        # Base foot positions in body frame with increased ground clearance
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            pos = v.copy()
            pos[2] += 0.03  # Add vertical offset to prevent penetration
            self.base_feet_pos_body[k] = pos
        
        # Diagonal pair phase offsets (FL+RR at 0.0, FR+RL at 0.5)
        self.phase_offsets = {
            leg_names[0]: 0.0,   # FL
            leg_names[1]: 0.5,   # FR
            leg_names[2]: 0.5,   # RL
            leg_names[3]: 0.0,   # RR
        }
        
        # Swing timing per leg (start_phase for each leg's swing window)
        self.swing_start = {
            leg_names[0]: 0.25,   # FL swings [0.25, 0.375]
            leg_names[1]: 0.125,  # FR swings [0.125, 0.25]
            leg_names[2]: 0.0,    # RL swings [0.0, 0.125]
            leg_names[3]: 0.375,  # RR swings [0.375, 0.5]
        }
        self.swing_duration = 0.125  # Each swing lasts 12.5% of phase
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Orbital motion parameters
        self.vx = 0.8          # Forward velocity magnitude
        self.yaw_rate = 1.2    # Constant yaw rate for circular path
        
        # Roll oscillation parameters (reduced for smoother motion)
        self.roll_rate_amp = 1.8  # Roll rate amplitude (reduced from 2.5)
        
        # Leg extension modulation for roll support
        self.radial_extension_gain = 0.08  # How much legs extend/retract with roll

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity, constant yaw rate,
        and sinusoidal roll rate that oscillates twice per orbit.
        Enforces ground contact by adjusting base height.
        """
        # Constant forward velocity in body x direction
        vx = self.vx
        
        # Roll rate oscillates twice per cycle with reduced amplitude
        roll_rate = -self.roll_rate_amp * np.sin(4 * np.pi * phase)
        
        # Constant yaw rate for circular orbit
        yaw_rate = self.yaw_rate
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Enforce ground contact: adjust base height to ensure at least one foot touches ground
        min_foot_z_world = float('inf')
        for leg_name in self.leg_names:
            foot_body = self.compute_foot_position_body_frame(leg_name, phase)
            foot_world = transform_point_body_to_world(foot_body, self.root_pos, self.root_quat)
            min_foot_z_world = min(min_foot_z_world, foot_world[2])
        
        # Adjust base height if lowest foot is above ground
        if min_foot_z_world > 0.0:
            self.root_pos[2] -= min_foot_z_world

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with:
        1. Diagonal gait swing/stance pattern with smooth transitions
        2. Radial extension modulation based on roll angle
        3. Vertical compensation for lateral extension
        """
        # Base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is in swing phase
        swing_start = self.swing_start[leg_name]
        swing_end = (swing_start + self.swing_duration) % 1.0
        
        # Compute normalized phase within cycle [0, 1)
        phase_norm = phase % 1.0
        
        # Check if in swing (handle wrap-around)
        if swing_start < swing_end:
            in_swing = swing_start <= phase_norm < swing_end
        else:
            in_swing = phase_norm >= swing_start or phase_norm < swing_end
        
        # Compute swing progress if in swing
        if in_swing:
            if swing_start < swing_end:
                swing_progress = (phase_norm - swing_start) / self.swing_duration
            else:
                if phase_norm >= swing_start:
                    swing_progress = (phase_norm - swing_start) / self.swing_duration
                else:
                    swing_progress = (phase_norm + 1.0 - swing_start) / self.swing_duration
            
            swing_progress = np.clip(swing_progress, 0.0, 1.0)
            
            # Use quintic polynomial for smooth swing trajectory (C2 continuous)
            # s(t) = 6t^5 - 15t^4 + 10t^3 (zero velocity and acceleration at endpoints)
            t = swing_progress
            smooth_curve = 6*t**5 - 15*t**4 + 10*t**3
            
            # Forward advance during swing
            foot[0] += self.step_length * (smooth_curve - 0.5)
            
            # Vertical arc with quintic smoothing
            # Use smooth_curve for vertical motion to ensure C2 continuity
            arc_height = np.sin(np.pi * smooth_curve)
            foot[2] += self.step_height * arc_height
            
        else:
            # Stance phase: compute stance progress properly
            stance_duration = 1.0 - self.swing_duration
            
            if swing_start < swing_end:
                # Normal case: swing doesn't wrap
                if phase_norm >= swing_end:
                    stance_progress = (phase_norm - swing_end) / stance_duration
                else:
                    stance_progress = (phase_norm + 1.0 - swing_end) / stance_duration
            else:
                # Swing wraps around: stance is between swing_end and swing_start
                stance_progress = (phase_norm - swing_end) / stance_duration
            
            stance_progress = np.clip(stance_progress, 0.0, 1.0)
            
            # Retract foot backward relative to body motion
            foot[0] -= self.step_length * stance_progress
        
        # Radial extension modulation based on current roll angle
        roll, _, _ = quat_to_euler(self.root_quat)
        
        # Determine if this is a left or right leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compute lateral extension
        if is_left_leg:
            extension = self.radial_extension_gain * roll
        else:
            extension = -self.radial_extension_gain * roll
        
        # Apply radial extension in body y direction (lateral)
        foot[1] += extension
        
        # Vertical compensation for lateral extension to maintain ground clearance
        # When foot extends laterally and base is rolled, adjust vertical position
        extension_magnitude = abs(extension)
        roll_magnitude = abs(roll)
        vertical_compensation = extension_magnitude * np.sin(roll_magnitude) * 0.5
        foot[2] += vertical_compensation
        
        return foot


def transform_point_body_to_world(point_body, root_pos, root_quat):
    """Helper function to transform a point from body frame to world frame."""
    # Rotate point by quaternion
    point_rotated = quat_rotate(root_quat, point_body)
    # Translate by root position
    point_world = root_pos + point_rotated
    return point_world


def quat_rotate(quat, vec):
    """Rotate a vector by a quaternion."""
    # Convert quaternion to rotation matrix and apply
    q = quat
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R @ vec