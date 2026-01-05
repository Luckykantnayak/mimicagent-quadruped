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
        self.roll_amplitude = 0.4  # radians (~23 degrees peak roll)
        self.roll_freq = 2.0  # two roll cycles per phase cycle
        
        # Gait parameters for diagonal trot
        self.step_length = 0.12
        self.step_height = 0.06
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
        
        # Extension modulation for roll support
        self.lateral_extension_amplitude = 0.03  # additional lateral extension for outer legs

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity and constant yaw rate for circular path,
        combined with sinusoidal roll rate for oscillating roll motion.
        
        Roll oscillation: roll_rate = -roll_amplitude * 2π * roll_freq * sin(2π * roll_freq * phase)
        This produces roll angle: roll = roll_amplitude * cos(2π * roll_freq * phase)
        
        At phase 0.0: roll = roll_amplitude (max left)
        At phase 0.25: roll = 0 (neutral)
        At phase 0.5: roll = -roll_amplitude (max right)
        At phase 0.75: roll = 0 (neutral)
        At phase 1.0: roll = roll_amplitude (back to max left)
        
        Actually, adjusting to match spec:
        - phase 0.25: max left roll
        - phase 0.5: max right roll
        - phase 0.75: max left roll
        - phase 0.0/1.0: neutral
        
        This requires: roll = roll_amplitude * sin(2π * roll_freq * phase) with roll_freq = 2
        roll_rate = roll_amplitude * 2π * roll_freq * cos(2π * roll_freq * phase)
        """
        
        # Constant forward velocity and yaw rate for circular path
        vx = self.forward_velocity
        yaw_rate = self.yaw_rate
        
        # Sinusoidal roll rate for oscillating roll motion
        # roll = roll_amplitude * sin(4π * phase)
        # roll_rate = roll_amplitude * 4π * cos(4π * phase)
        roll_rate = self.roll_amplitude * 2.0 * np.pi * self.roll_freq * np.cos(2.0 * np.pi * self.roll_freq * phase)
        
        # Set velocities in world frame
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
            
            # Vertical arc (parabolic trajectory)
            swing_height = self.step_height * np.sin(np.pi * progress)
            foot[2] += swing_height
        
        # Lateral extension modulation based on roll angle
        # Current roll angle from integrated orientation
        roll, _, _ = quat_to_euler(self.root_quat)
        
        # Outer legs (relative to roll direction) extend more laterally
        # Left roll (positive roll): FL and RL extend left (negative y), FR and RR retract
        # Right roll (negative roll): FR and RR extend right (positive y), FL and RL retract
        
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            # Left side legs: extend more when roll is positive (left roll)
            lateral_adjustment = -self.lateral_extension_amplitude * roll
        else:  # FR or RR
            # Right side legs: extend more when roll is negative (right roll)
            lateral_adjustment = self.lateral_extension_amplitude * roll
        
        foot[1] += lateral_adjustment
        
        return foot