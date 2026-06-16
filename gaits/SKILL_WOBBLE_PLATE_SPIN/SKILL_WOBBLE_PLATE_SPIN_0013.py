from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_WOBBLE_PLATE_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Wobble Plate Spin: Continuous in-place yaw rotation (360 degrees per cycle)
    with superimposed wobbling motion created by coupled pitch and roll oscillations.
    
    - Base executes constant yaw rotation (360 deg/cycle)
    - Pitch and roll oscillate sinusoidally with phase offsets
    - All four feet maintain near-continuous ground contact with micro-lifts during peak tilts
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
        self.wobble_amplitude_pitch = 0.12  # radians (~6.9 degrees) - reduced for safety
        self.wobble_amplitude_roll = 0.12   # radians (~6.9 degrees) - reduced for safety
        self.wobble_freq = 2.0  # 2 complete wobble cycles per phase cycle (4 quadrants)
        
        # Foot adjustment scaling for maintaining ground contact
        self.foot_adjust_scale = 0.6  # increased from 0.3 for better compensation
        
        # Micro-lift parameters for unloaded legs during peak tilts
        self.lift_threshold = 0.06  # radians - tilt angle threshold to trigger micro-lift
        self.max_lift_height = 0.03  # meters - maximum lift for unloaded legs
        
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
        wobble_phase = 2.0 * np.pi * self.wobble_freq * phase
        
        # Pitch rate: derivative of pitch angle
        pitch_rate = self.wobble_amplitude_pitch * self.wobble_freq * 2.0 * np.pi * np.cos(wobble_phase)
        
        # Roll rate: 90-degree phase offset to create quadrant pattern
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
        - Apply micro-lift to unloaded legs during peak tilts to prevent penetration
        
        Pitch compensation: forward pitch -> rear feet extend back, front feet extend forward
        Roll compensation: right roll -> left feet extend left, right feet extend right
        Z compensation: CORRECTED - lift feet appropriately based on tilt geometry
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute current wobble angles (integrated from rates)
        wobble_phase = 2.0 * np.pi * self.wobble_freq * phase
        pitch_angle = self.wobble_amplitude_pitch * np.sin(wobble_phase)
        roll_angle = self.wobble_amplitude_roll * np.cos(wobble_phase)
        
        # Determine leg position (front/rear, left/right)
        is_front = leg_name.startswith('F')
        is_left = leg_name.endswith('L')
        
        # Pitch compensation (x-direction in body frame)
        # Forward pitch (positive): front legs extend forward, rear legs extend back
        if is_front:
            foot[0] += self.foot_adjust_scale * pitch_angle
        else:
            foot[0] -= self.foot_adjust_scale * pitch_angle
        
        # Roll compensation (y-direction in body frame)
        # Right roll (positive): left legs extend left, right legs extend right
        if is_left:
            foot[1] -= self.foot_adjust_scale * roll_angle
        else:
            foot[1] += self.foot_adjust_scale * roll_angle
        
        # Z compensation: CORRECTED LOGIC
        # When body tilts, the "high" side needs feet to lift (less negative Z in body frame)
        # The "low" side is weight-bearing and should maintain or slightly lower
        
        # Base Z adjustment for tilt geometry (lift all feet slightly during tilt)
        tilt_magnitude = np.sqrt(pitch_angle**2 + roll_angle**2)
        base_z_lift = 0.4 * self.foot_adjust_scale * tilt_magnitude  # CORRECTED: now adds (lifts)
        foot[2] += base_z_lift
        
        # Micro-lift for unloaded legs during peak tilts
        # This prevents geometric overconstrain when all four legs try to stay grounded during tilt
        
        # Roll-based micro-lift
        if abs(roll_angle) > self.lift_threshold:
            if roll_angle > 0:  # Rolling right - left legs unload
                if is_left:
                    # Smooth lift proportional to how far past threshold
                    lift_factor = (abs(roll_angle) - self.lift_threshold) / (self.wobble_amplitude_roll - self.lift_threshold)
                    lift_factor = np.clip(lift_factor, 0.0, 1.0)
                    foot[2] += self.max_lift_height * lift_factor
            else:  # Rolling left - right legs unload
                if not is_left:
                    lift_factor = (abs(roll_angle) - self.lift_threshold) / (self.wobble_amplitude_roll - self.lift_threshold)
                    lift_factor = np.clip(lift_factor, 0.0, 1.0)
                    foot[2] += self.max_lift_height * lift_factor
        
        # Pitch-based micro-lift
        if abs(pitch_angle) > self.lift_threshold:
            if pitch_angle > 0:  # Pitching forward - front legs unload
                if is_front:
                    lift_factor = (abs(pitch_angle) - self.lift_threshold) / (self.wobble_amplitude_pitch - self.lift_threshold)
                    lift_factor = np.clip(lift_factor, 0.0, 1.0)
                    foot[2] += self.max_lift_height * lift_factor * 0.5  # Reduced effect for pitch
            else:  # Pitching backward - rear legs unload
                if not is_front:
                    lift_factor = (abs(pitch_angle) - self.lift_threshold) / (self.wobble_amplitude_pitch - self.lift_threshold)
                    lift_factor = np.clip(lift_factor, 0.0, 1.0)
                    foot[2] += self.max_lift_height * lift_factor * 0.5  # Reduced effect for pitch
        
        return foot