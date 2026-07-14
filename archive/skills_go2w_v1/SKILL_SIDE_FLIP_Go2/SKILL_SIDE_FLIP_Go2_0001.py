from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_SIDE_FLIP_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Lateral acrobatic flip: robot executes a 360° roll rotation while airborne.
    
    Phases:
      0.0 - 0.25: Crouch and weight shift
      0.25 - 0.4: Launch with asymmetric thrust (left > right) to generate roll momentum
      0.4 - 0.75: Aerial rotation (all feet off ground, 360° roll)
      0.75 - 0.85: Landing preparation (legs extend for touchdown)
      0.85 - 1.0: Impact absorption (legs flex to dissipate energy)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.25  # ~0.8s per flip cycle
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.crouch_depth = 0.08  # Vertical retraction during crouch
        self.launch_extension = 0.12  # Leg extension magnitude during launch
        self.launch_asymmetry = 0.20  # Left legs extend 20% more than right for roll
        self.tuck_amount = 0.06  # Leg retraction during aerial phase
        self.landing_extension = 0.10  # Leg reach during landing prep
        self.absorption_depth = 0.07  # Leg flexion during impact absorption
        
        # Roll dynamics
        self.target_roll_rate = 1100.0  # deg/s sustained during aerial phase
        self.launch_vz = 1.8  # Upward velocity magnitude during launch (m/s)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Classify legs into left and right groups
        self.left_legs = [leg for leg in leg_names if leg.startswith('FL') or leg.startswith('RL')]
        self.right_legs = [leg for leg in leg_names if leg.startswith('FR') or leg.startswith('RR')]

    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent velocities and angular rates.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0 - 0.25: Crouch (lowering base)
        if phase < 0.25:
            # Smooth downward velocity during crouch
            progress = phase / 0.25
            vz = -0.5 * (1.0 - np.cos(np.pi * progress))  # Smooth bell curve downward
            
        # Phase 0.25 - 0.4: Launch (explosive upward + roll initiation)
        elif phase < 0.4:
            progress = (phase - 0.25) / 0.15
            # Strong upward velocity
            vz = self.launch_vz * np.sin(np.pi * progress)
            # Initiate roll rate (ramp up quickly)
            roll_rate = np.deg2rad(self.target_roll_rate) * (progress ** 0.5)
            # Small lateral velocity from asymmetry
            vy = 0.15 * progress
            
        # Phase 0.4 - 0.75: Aerial rotation (ballistic trajectory + sustained roll)
        elif phase < 0.75:
            progress = (phase - 0.4) / 0.35
            # Ballistic vertical velocity (parabolic: up then down)
            time_in_flight = progress * 0.35 / self.freq
            vz = self.launch_vz * 0.6 - 9.81 * time_in_flight * 1.5
            # Sustained roll rate from angular momentum conservation
            roll_rate = np.deg2rad(self.target_roll_rate)
            # Residual lateral velocity
            vy = 0.15 * (1.0 - progress * 0.5)
            
        # Phase 0.75 - 0.85: Landing preparation (descending, roll rate decreasing)
        elif phase < 0.85:
            progress = (phase - 0.75) / 0.1
            # Continuing downward
            vz = -1.2 - 0.8 * progress
            # Roll rate decaying as rotation completes
            roll_rate = np.deg2rad(self.target_roll_rate) * (1.0 - progress)
            # Lateral velocity decaying
            vy = 0.075 * (1.0 - progress)
            
        # Phase 0.85 - 1.0: Impact absorption (deceleration to rest)
        else:
            progress = (phase - 0.85) / 0.15
            # Rapid deceleration from impact absorption
            vz = -2.0 * (1.0 - progress) ** 2
            # Roll rate to zero
            roll_rate = 0.0
            vy = 0.0
        
        # Set velocities and integrate pose
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
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
        
        Left legs (FL, RL) extend more during launch to create roll moment.
        Right legs (FR, RR) extend less during launch.
        All legs symmetric during aerial, landing prep, and absorption.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name in self.left_legs
        
        # Phase 0.0 - 0.25: Crouch (retract vertically)
        if phase < 0.25:
            progress = phase / 0.25
            # Smooth retraction upward in body frame (Z increases = leg retracts)
            retraction = self.crouch_depth * np.sin(np.pi * progress / 2.0)
            foot[2] += retraction
            # Slight inward motion (reduce stance width)
            foot[1] *= (1.0 - 0.1 * progress)
            
        # Phase 0.25 - 0.4: Launch (asymmetric extension)
        elif phase < 0.4:
            progress = (phase - 0.25) / 0.15
            # Start from crouched position
            foot[2] += self.crouch_depth
            # Extend downward (Z decreases)
            extension = self.launch_extension * np.sin(np.pi * progress)
            if is_left_leg:
                # Left legs extend more
                extension *= (1.0 + self.launch_asymmetry)
            else:
                # Right legs extend less
                extension *= (1.0 - self.launch_asymmetry * 0.5)
            foot[2] -= extension
            
        # Phase 0.4 - 0.75: Aerial rotation (tucked legs)
        elif phase < 0.75:
            progress = (phase - 0.4) / 0.35
            # Legs tucked toward body (retracted upward)
            tuck = self.tuck_amount * (1.0 - abs(2.0 * progress - 1.0))  # Peak tuck mid-flight
            foot[2] += tuck
            # Slight inward tuck
            foot[0] *= (1.0 - 0.05 * (1.0 - abs(2.0 * progress - 1.0)))
            foot[1] *= (1.0 - 0.1 * (1.0 - abs(2.0 * progress - 1.0)))
            
        # Phase 0.75 - 0.85: Landing preparation (extend for contact)
        elif phase < 0.85:
            progress = (phase - 0.75) / 0.1
            # Extend legs downward smoothly
            extension = self.landing_extension * progress
            foot[2] -= extension
            # Widen stance slightly for stability
            foot[1] *= (1.0 + 0.08 * progress)
            
        # Phase 0.85 - 1.0: Impact absorption (flex legs)
        else:
            progress = (phase - 0.85) / 0.15
            # Start from extended landing position
            foot[2] -= self.landing_extension
            # Flex upward to absorb impact
            absorption = self.absorption_depth * np.sin(np.pi * progress / 2.0)
            foot[2] += absorption
            # Return to neutral stance width
            foot[1] *= (1.0 + 0.08 * (1.0 - progress))
        
        return foot