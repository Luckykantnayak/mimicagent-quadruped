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
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
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
        
        # Roll oscillation parameters
        self.roll_rate_amp = 2.5  # Roll rate amplitude (rad/s effective scale)
        
        # Leg extension modulation for roll support
        self.radial_extension_gain = 0.08  # How much legs extend/retract with roll

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity, constant yaw rate,
        and sinusoidal roll rate that oscillates twice per orbit.
        
        Roll rate = A * sin(4π * phase)
        - Peaks at phase 0.25 (negative, left roll) and 0.75 (positive, right roll)
        - Zero at phase 0.0, 0.5, 1.0
        """
        # Constant forward velocity in body x direction
        vx = self.vx
        
        # Roll rate oscillates twice per cycle: sin(4π * phase)
        # At phase 0.25: sin(π) peaks negative → rolling left
        # At phase 0.5: sin(2π) = 0 → neutral
        # At phase 0.75: sin(3π) peaks positive → rolling right
        # At phase 1.0: sin(4π) = 0 → neutral
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

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with:
        1. Diagonal gait swing/stance pattern
        2. Radial extension modulation based on roll angle
        
        Outer legs (relative to roll direction) extend radially.
        Inner legs retract radially.
        """
        # Base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is in swing phase
        swing_start = self.swing_start[leg_name]
        swing_end = (swing_start + self.swing_duration) % 1.0
        
        # Check if in swing (handle wrap-around)
        if swing_start < swing_end:
            in_swing = swing_start <= phase < swing_end
        else:
            in_swing = phase >= swing_start or phase < swing_end
        
        # Swing/stance trajectory
        if in_swing:
            # Swing phase: advance foot forward with arc
            if swing_start < swing_end:
                swing_progress = (phase - swing_start) / self.swing_duration
            else:
                if phase >= swing_start:
                    swing_progress = (phase - swing_start) / self.swing_duration
                else:
                    swing_progress = (phase + 1.0 - swing_start) / self.swing_duration
            
            # Forward advance during swing
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # Vertical arc clearance
            swing_angle = np.pi * swing_progress
            foot[2] += self.step_height * np.sin(swing_angle)
        else:
            # Stance phase: retract foot backward relative to body motion
            if swing_start < swing_end:
                if phase < swing_start:
                    stance_progress = (phase + 1.0 - swing_end) / (1.0 - self.swing_duration)
                else:
                    stance_progress = (phase - swing_end) / (1.0 - self.swing_duration)
            else:
                stance_progress = (phase - swing_end) / (1.0 - self.swing_duration)
            
            foot[0] -= self.step_length * stance_progress
        
        # Radial extension modulation based on current roll angle
        # Extract current roll from quaternion
        roll, _, _ = quat_to_euler(self.root_quat)
        
        # Determine if this is a left or right leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Outer legs extend when roll tilts away from them
        # Left legs extend when roll > 0 (tilted right), retract when roll < 0 (tilted left)
        # Right legs extend when roll < 0 (tilted left), retract when roll > 0 (tilted right)
        if is_left_leg:
            # Left leg: extend when rolling right (positive roll)
            extension = self.radial_extension_gain * roll
        else:
            # Right leg: extend when rolling left (negative roll)
            extension = -self.radial_extension_gain * roll
        
        # Apply radial extension in body y direction (lateral)
        foot[1] += extension
        
        return foot