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
        self.freq = 1.25
        
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Reduced motion parameters to respect kinematic limits
        self.crouch_depth = 0.04  # Reduced from 0.08 to avoid extreme flexion
        self.launch_extension = 0.08  # Reduced from 0.12 to stay within joint limits
        self.launch_asymmetry = 0.15  # Reduced from 0.20 for more balanced ground clearance
        self.tuck_amount = 0.03  # Reduced from 0.06 to avoid extreme retraction
        self.landing_extension = 0.15  # Increased from 0.10 to ensure ground reach
        self.absorption_depth = 0.04  # Reduced from 0.07 to limit flexion demands
        
        # Adjusted roll dynamics
        self.target_roll_rate = 1100.0
        self.launch_vz = 2.4  # Increased from 1.8 to maintain altitude during aerial phase
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.left_legs = [leg for leg in leg_names if leg.startswith('FL') or leg.startswith('RL')]
        self.right_legs = [leg for leg in leg_names if leg.startswith('FR') or leg.startswith('RR')]

    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent velocities with altitude awareness.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0 - 0.25: Crouch (mild lowering)
        if phase < 0.25:
            progress = phase / 0.25
            # Gentle downward velocity with smooth envelope
            vz = -0.3 * np.sin(np.pi * progress)
            
        # Phase 0.25 - 0.4: Launch (strong upward thrust with roll initiation)
        elif phase < 0.4:
            progress = (phase - 0.25) / 0.15
            # Strong upward velocity with smooth ramp
            vz = self.launch_vz * np.sin(np.pi * progress)
            # Roll rate ramps up smoothly
            roll_rate = np.deg2rad(self.target_roll_rate) * np.sin(np.pi * progress / 2.0)
            # Lateral velocity from asymmetric thrust
            vy = 0.12 * np.sin(np.pi * progress / 2.0)
            
        # Phase 0.4 - 0.75: Aerial rotation (realistic ballistic trajectory)
        elif phase < 0.75:
            progress = (phase - 0.4) / 0.35
            # Ballistic vertical velocity: parabolic descent with realistic gravity
            # Time since launch apex
            aerial_time = progress * 0.35 / self.freq
            # Parabolic velocity profile peaking early then descending
            vz = self.launch_vz * 0.7 * (1.0 - 2.0 * progress) - 9.81 * aerial_time * 0.8
            # Sustained roll rate from angular momentum
            roll_rate = np.deg2rad(self.target_roll_rate) * (0.5 + 0.5 * np.cos(2.0 * np.pi * progress))
            # Lateral velocity decaying
            vy = 0.12 * (1.0 - progress * 0.6)
            
        # Phase 0.75 - 0.85: Landing preparation (controlled descent)
        elif phase < 0.85:
            progress = (phase - 0.75) / 0.1
            # Controlled downward velocity ramping to contact
            vz = -0.8 * (1.0 + progress)
            # Roll rate decaying to complete 360 degrees
            roll_rate = np.deg2rad(self.target_roll_rate) * 0.4 * (1.0 - progress) ** 2
            # Lateral velocity minimal
            vy = 0.05 * (1.0 - progress)
            
        # Phase 0.85 - 1.0: Impact absorption (deceleration with altitude protection)
        else:
            progress = (phase - 0.85) / 0.15
            # Smooth deceleration to rest
            vz = -1.5 * (1.0 - progress) ** 3
            # Roll stabilizes to zero
            roll_rate = 0.0
            vy = 0.0
        
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
        Compute foot position in body frame with conservative kinematic envelope.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name in self.left_legs
        
        # Phase 0.0 - 0.25: Crouch (mild retraction)
        if phase < 0.25:
            progress = phase / 0.25
            # Smooth retraction with reduced amplitude
            retraction = self.crouch_depth * np.sin(np.pi * progress / 2.0)
            foot[2] += retraction
            # Minimal lateral adjustment
            foot[1] *= (1.0 - 0.05 * progress)
            
        # Phase 0.25 - 0.4: Launch (asymmetric extension)
        elif phase < 0.4:
            progress = (phase - 0.25) / 0.15
            # Start from crouched position
            foot[2] += self.crouch_depth
            # Extend downward with asymmetry for roll generation
            extension = self.launch_extension * np.sin(np.pi * progress)
            if is_left_leg:
                # Left legs extend more to create roll moment
                extension *= (1.0 + self.launch_asymmetry)
            else:
                # Right legs extend less but still maintain ground clearance
                extension *= (1.0 - self.launch_asymmetry * 0.3)
            foot[2] -= extension
            
        # Phase 0.4 - 0.75: Aerial rotation (minimal tuck)
        elif phase < 0.75:
            progress = (phase - 0.4) / 0.35
            # Mild tuck during peak altitude only
            tuck_envelope = np.sin(np.pi * progress)
            tuck = self.tuck_amount * tuck_envelope
            foot[2] += tuck
            # Reduced horizontal contraction
            contraction_factor = 0.03 * tuck_envelope
            foot[0] *= (1.0 - contraction_factor)
            foot[1] *= (1.0 - contraction_factor * 2.0)
            
        # Phase 0.75 - 0.85: Landing preparation (extended reach)
        elif phase < 0.85:
            progress = (phase - 0.75) / 0.1
            # Strong extension to ensure ground contact
            extension = self.landing_extension * np.sin(np.pi * progress / 2.0)
            foot[2] -= extension
            # Widen stance for stability
            foot[1] *= (1.0 + 0.1 * progress)
            
        # Phase 0.85 - 1.0: Impact absorption (controlled flexion)
        else:
            progress = (phase - 0.85) / 0.15
            # Start from extended landing position
            foot[2] -= self.landing_extension
            # Absorb impact with controlled upward motion
            absorption = self.absorption_depth * np.sin(np.pi * progress / 2.0)
            foot[2] += absorption
            # Return to neutral stance
            foot[1] *= (1.0 + 0.1 * (1.0 - progress))
        
        return foot