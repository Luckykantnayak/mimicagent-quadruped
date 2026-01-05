from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ORBIT_TILT_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Orbital motion with synchronized roll oscillation.
    
    - Robot travels in continuous circular path (constant forward velocity + constant yaw rate)
    - Base executes synchronized roll oscillation (two full left-right cycles per orbit)
    - Diagonal trotting gait with outer legs extending more during roll
    - Roll peaks occur at cardinal orbital directions (phase 0.25, 0.5, 0.75)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        # Initialize base class
        BaseMotionGenerator.__init__(self, initial_foot_positions_body, freq=1.0)
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Circular motion parameters
        self.forward_velocity = 0.5  # constant forward velocity
        self.yaw_rate = 1.0  # constant yaw rate to produce circular path
        
        # Roll oscillation parameters
        # Two complete roll cycles per orbit (left-right-left-right)
        self.roll_amplitude = 0.25  # radians (~14 degrees peak roll, reduced from 0.4)
        self.roll_freq = 2.0  # two roll cycles per phase cycle
        
        # Base height management
        self.nominal_base_height = 0.32  # nominal standing height
        self.height_compensation_factor = 0.15  # additional height per radian of roll magnitude
        
        # Gait parameters for diagonal trot
        self.step_length = 0.09  # reduced from 0.12 to avoid workspace limits
        self.step_height = 0.05  # slightly reduced for safer clearance
        self.duty_cycle = 0.6
        
        # Phase offsets for diagonal trot coordination
        # FL-RR: group 1 (offset 0.0)
        # FR-RL: group 2 (offset 0.5)
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0
            else:  # FR or RL
                self.phase_offsets[leg] = 0.5
        
        # Extension modulation for roll support (reduced from 0.03)
        self.lateral_extension_amplitude = 0.012
        
        # Smoothing for lateral extension (temporal filtering)
        self.filtered_roll = 0.0
        self.roll_filter_alpha = 0.15  # low-pass filter coefficient

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity and constant yaw rate for circular path,
        combined with sinusoidal roll rate for oscillating roll motion.
        
        Roll oscillation: roll = roll_amplitude * sin(2π * roll_freq * phase)
        roll_rate = roll_amplitude * 2π * roll_freq * cos(2π * roll_freq * phase)
        
        Base height dynamically adjusts to compensate for roll-induced geometric height loss.
        """
        
        # Constant forward velocity and yaw rate for circular path
        vx = self.forward_velocity
        yaw_rate = self.yaw_rate
        
        # Sinusoidal roll rate for oscillating roll motion
        roll_rate = self.roll_amplitude * 2.0 * np.pi * self.roll_freq * np.cos(2.0 * np.pi * self.roll_freq * phase)
        
        # Compute target roll angle from phase for height compensation
        target_roll = self.roll_amplitude * np.sin(2.0 * np.pi * self.roll_freq * phase)
        
        # Dynamic base height to compensate for roll-induced geometric lowering
        target_height = self.nominal_base_height + self.height_compensation_factor * abs(target_roll)
        
        # Compute vertical velocity to smoothly track target height
        current_height = self.root_pos[2]
        height_error = target_height - current_height
        vz = 3.0 * height_error  # proportional controller with gain 3.0
        
        # Set velocities in world frame
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([roll_rate, 0.0, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Update filtered roll for smooth lateral extension (low-pass filter)
        roll_current, _, _ = quat_to_euler(self.root_quat)
        self.filtered_roll = self.roll_filter_alpha * roll_current + (1.0 - self.roll_filter_alpha) * self.filtered_roll

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        
        Uses diagonal trot gait pattern with lateral extension modulation
        to support roll oscillation (outer legs extend more during roll).
        """
        
        # Compute leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start with base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is in stance or swing
        if leg_phase < self.duty_cycle:
            # STANCE PHASE
            # Foot moves rearward in body frame as base advances
            progress = leg_phase / self.duty_cycle
            foot[0] -= self.step_length * (progress - 0.5)
            
        else:
            # SWING PHASE
            # Foot lifts, moves forward with arc trajectory
            progress = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)
            
            # Forward motion during swing
            foot[0] += self.step_length * (progress - 0.5)
            
            # Vertical arc (parabolic trajectory with smooth blending)
            swing_height = self.step_height * np.sin(np.pi * progress)
            foot[2] += swing_height
        
        # Lateral extension modulation based on filtered roll angle
        # Use filtered roll for smoother acceleration profile
        roll = self.filtered_roll
        
        # Apply saturation to prevent extreme lateral offsets
        # tanh provides smooth saturation near limits
        roll_normalized = np.tanh(roll / self.roll_amplitude)
        
        # Outer legs (relative to roll direction) extend more laterally
        # Left roll (positive roll): FL and RL extend left (negative y), FR and RR retract
        # Right roll (negative roll): FR and RR extend right (positive y), FL and RL retract
        
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            # Left side legs: extend more when roll is positive (left roll)
            lateral_adjustment = -self.lateral_extension_amplitude * roll_normalized
        else:  # FR or RR
            # Right side legs: extend more when roll is negative (right roll)
            lateral_adjustment = self.lateral_extension_amplitude * roll_normalized
        
        foot[1] += lateral_adjustment
        
        return foot