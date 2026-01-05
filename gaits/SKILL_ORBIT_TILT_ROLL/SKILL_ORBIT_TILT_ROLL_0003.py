from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ORBIT_TILT_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Orbit with synchronized roll oscillation.
    
    Robot travels in a continuous circular path while executing rhythmic roll oscillations.
    Base rolls left-neutral-right-neutral cyclically, with outer legs extending to support
    the tilt while maintaining forward orbital progression.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        # Initialize base class
        BaseMotionGenerator.__init__(self, initial_foot_positions_body, freq=0.5)
        
        self.leg_names = leg_names
        
        # Motion parameters
        self.freq = 0.5  # Complete one orbit + full roll cycle per period
        
        # Base motion parameters for circular trajectory
        self.forward_velocity = 0.8  # m/s forward velocity
        self.yaw_rate = 1.5  # rad/s to create circular path
        
        # Roll oscillation parameters (reduced from 20 to 12 degrees for workspace constraints)
        self.max_roll_angle = np.deg2rad(12)
        
        # Leg extension parameters (reduced to respect joint limits)
        self.lateral_extension_max = 0.05  # Maximum lateral extension for outer legs (m)
        self.vertical_extension_max = 0.025  # Maximum vertical extension for outer legs (m)
        
        # Ground contact parameters
        self.stance_height = -0.30  # Nominal z-position of feet in body frame for ground contact
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Identify left and right legs
        self.left_legs = [leg for leg in leg_names if leg.startswith('FL') or leg.startswith('RL')]
        self.right_legs = [leg for leg in leg_names if leg.startswith('FR') or leg.startswith('RR')]
        
        # Track current roll angle for ground compensation
        self.current_roll = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base motion: constant forward velocity with constant yaw rate for circular path,
        plus sinusoidal roll oscillation synchronized to phase.
        """
        
        # Constant forward velocity and yaw rate for circular trajectory
        vx = self.forward_velocity
        yaw_rate = self.yaw_rate
        
        # Smooth roll profile using sinusoidal oscillation
        # roll(phase) = max_roll * sin(2π * phase)
        # This creates smooth transitions through neutral at phase=0, 0.5, 1.0
        # Peak left at phase=0.25, peak right at phase=0.75
        self.current_roll = self.max_roll_angle * np.sin(2 * np.pi * phase)
        
        # Roll rate is derivative: d(roll)/dt = max_roll * 2π * freq * cos(2π * phase)
        roll_rate = self.max_roll_angle * 2 * np.pi * self.freq * np.cos(2 * np.pi * phase)
        
        # Set velocities
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
        Compute foot position in body frame with lateral and vertical extensions
        based on roll phase. Maintains ground contact by compensating for body roll.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left or right leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compute smooth extension factor using quintic smoothstep for C2 continuity
        # Left legs extend more when body rolls right (phase 0.5-1.0)
        # Right legs extend more when body rolls left (phase 0.0-0.5)
        
        if is_left_leg:
            # Left legs are outer during right roll: phase 0.5 to 1.0
            # Create smooth wave that peaks at phase 0.75
            phase_shifted = (phase + 0.25) % 1.0  # Shift so peak is at 0.5 in shifted coordinates
            if phase_shifted < 0.5:
                # Rising from 0 to 1
                t = phase_shifted * 2.0  # Map [0, 0.5] to [0, 1]
                extension_factor = self._smoothstep(t)
            else:
                # Falling from 1 to 0
                t = (phase_shifted - 0.5) * 2.0  # Map [0.5, 1.0] to [0, 1]
                extension_factor = 1.0 - self._smoothstep(t)
        else:
            # Right legs are outer during left roll: phase 0.0 to 0.5
            # Create smooth wave that peaks at phase 0.25
            phase_shifted = (phase - 0.25) % 1.0  # Shift so peak is at 0.5 in shifted coordinates
            if phase_shifted < 0.5:
                # Rising from 0 to 1
                t = phase_shifted * 2.0
                extension_factor = self._smoothstep(t)
            else:
                # Falling from 1 to 0
                t = (phase_shifted - 0.5) * 2.0
                extension_factor = 1.0 - self._smoothstep(t)
        
        # Apply lateral extension (y-direction in body frame)
        lateral_sign = 1.0 if is_left_leg else -1.0
        foot[1] += lateral_sign * self.lateral_extension_max * extension_factor
        
        # Apply vertical extension with ground contact compensation
        # Base vertical adjustment for extension
        vertical_adjustment = self.vertical_extension_max * extension_factor
        
        # Ground contact compensation: adjust foot height based on roll angle
        # When body rolls, the downward side needs less extension, upward side needs more
        # to maintain ground contact in world frame
        roll_compensation = 0.0
        if abs(self.current_roll) > 0.001:
            # Compute lateral distance from body center
            lateral_distance = foot[1]  # y-coordinate in body frame
            # When rolled, the vertical offset in world frame is approximately lateral_distance * sin(roll)
            # We need to compensate in body frame by approximately lateral_distance * tan(roll)
            # For small angles, tan(roll) ≈ roll, but use full formula for accuracy
            roll_compensation = -lateral_distance * np.tan(self.current_roll)
        
        # Apply vertical adjustment with roll compensation
        foot[2] = self.stance_height - vertical_adjustment + roll_compensation
        
        return foot
    
    def _smoothstep(self, t):
        """
        Quintic smoothstep function for C2 continuous interpolation.
        Maps [0, 1] to [0, 1] with zero first and second derivatives at boundaries.
        """
        t = np.clip(t, 0.0, 1.0)
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)