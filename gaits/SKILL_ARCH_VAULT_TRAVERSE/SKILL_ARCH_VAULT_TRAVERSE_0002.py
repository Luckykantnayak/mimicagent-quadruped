from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ARCH_VAULT_TRAVERSE_MotionGenerator(BaseMotionGenerator):
    """
    Arch vault traverse: A continuous forward locomotion pattern with alternating
    rear-leg-driven thrust phases and front-leg-dominated landing phases.
    
    The robot rocks through a four-phase cycle:
    - Phase 0.0-0.25: Rear compression, front extension, backward pitch preparation
    - Phase 0.25-0.5: Explosive rear thrust, flight phase, forward pitch
    - Phase 0.5-0.75: Front landing and support, rear swing forward, continued forward pitch
    - Phase 0.75-1.0: Rear touchdown, all legs contact, pitch stabilization
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.7  # Reduced frequency to allow smoother transitions
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - reduced to stay within kinematic limits
        self.rear_compression_depth = 0.05  # Reduced compression to minimize ground penetration risk
        self.front_extension_length = 0.10  # Reduced forward reach
        self.rear_swing_height = 0.12  # Reduced clearance for kinematic feasibility
        self.front_swing_height = 0.06  # Reduced front leg clearance
        self.vault_step_length = 0.16  # Reduced forward displacement per cycle
        
        # Base velocity parameters - moderated for smoother motion
        self.thrust_vx_peak = 0.9
        self.thrust_vz_peak = 0.5
        self.landing_vx = 0.5
        self.landing_vz = -0.3
        
        # Pitch control parameters - smoothed rates
        self.pitch_rate_back = -1.0
        self.pitch_rate_forward = 1.8
        self.pitch_rate_stabilize = -0.8
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Ensure initial base height provides clearance
        self.base_height_offset = 0.15

    def update_base_motion(self, phase, dt):
        """
        Update base velocity and angular velocity based on phase.
        Coordinate vertical motion to prevent ground penetration.
        """
        vx = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        if phase < 0.25:
            # Phase 0.0-0.25: Rear compression requires base to lift to maintain foot contact
            progress = smooth_step(phase / 0.25)
            vx = 0.15 * progress
            # Lift base to compensate for rear leg compression in body frame
            vz = 0.25 * np.sin(np.pi * progress)
            pitch_rate = self.pitch_rate_back * np.sin(np.pi * progress) * (1.0 - 0.3 * progress)
            
        elif phase < 0.5:
            # Phase 0.25-0.5: Explosive thrust and flight
            progress = smooth_step((phase - 0.25) / 0.25)
            vx = self.thrust_vx_peak * (0.5 + 0.5 * progress)
            vz = self.thrust_vz_peak * np.sin(np.pi * progress * 0.8)
            pitch_rate = self.pitch_rate_forward * (1.0 - 0.4 * progress)
            
        elif phase < 0.75:
            # Phase 0.5-0.75: Front landing, must descend smoothly
            progress = smooth_step((phase - 0.5) / 0.25)
            vx = self.landing_vx * (1.0 - 0.3 * progress)
            vz = self.landing_vz * np.sin(np.pi * progress * 0.6)
            pitch_rate = self.pitch_rate_forward * (0.6 - 0.5 * progress) * (1.0 - progress)
            
        else:
            # Phase 0.75-1.0: Rear touchdown and transition
            progress = smooth_step((phase - 0.75) / 0.25)
            vx = 0.25 * (1.0 - progress)
            vz = 0.05 * np.sin(np.pi * progress)
            pitch_rate = self.pitch_rate_stabilize * np.sin(np.pi * progress * 0.5)
        
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on leg name and phase.
        Ensures ground clearance and joint limit compliance.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_name.startswith('FL') or leg_name.startswith('FR'):
            # Front legs
            foot = self._compute_front_leg_trajectory(foot, phase)
        else:
            # Rear legs (RL, RR)
            foot = self._compute_rear_leg_trajectory(foot, phase)
        
        return foot
    
    def _compute_front_leg_trajectory(self, foot_base, phase):
        """
        Front leg trajectory through vault cycle.
        Maintains ground contact during landing and avoids downward offsets.
        """
        foot = foot_base.copy()
        
        if phase < 0.25:
            # Phase 0.0-0.25: Extend forward during rear compression
            progress = smooth_step(phase / 0.25)
            foot[0] += self.front_extension_length * progress
            foot[2] += 0.01 * np.sin(np.pi * progress * 0.5)  # Slight lift during extension
            
        elif phase < 0.5:
            # Phase 0.25-0.5: Lift off during flight with smooth swing
            progress = smooth_step((phase - 0.25) / 0.25)
            foot[0] += self.front_extension_length * (1.0 - 0.4 * progress)
            foot[2] += self.front_swing_height * np.sin(np.pi * progress)
            
        elif phase < 0.75:
            # Phase 0.5-0.75: Land and sweep backward in body frame
            progress = smooth_step((phase - 0.5) / 0.25)
            foot[0] += self.front_extension_length * (0.6 - 0.6 * progress)
            # Remove downward compression - maintain ground contact height
            foot[2] += 0.0
            
        else:
            # Phase 0.75-1.0: Transition to next cycle with smooth retraction
            progress = smooth_step((phase - 0.75) / 0.25)
            foot[0] += self.front_extension_length * progress * 0.3
            foot[2] += 0.0
            
        return foot
    
    def _compute_rear_leg_trajectory(self, foot_base, phase):
        """
        Rear leg trajectory through vault cycle.
        Reduces amplitudes to stay within joint limits and prevent ground penetration.
        """
        foot = foot_base.copy()
        
        if phase < 0.25:
            # Phase 0.0-0.25: Compress for thrust loading
            # Reduce Z motion in body frame since base will lift
            progress = smooth_step(phase / 0.25)
            foot[2] -= self.rear_compression_depth * np.sin(np.pi * progress)
            foot[0] += 0.02 * progress
            
        elif phase < 0.5:
            # Phase 0.25-0.5: Explosive extension and liftoff, begin forward swing
            progress = smooth_step((phase - 0.25) / 0.25)
            if progress < 0.5:
                # Extension phase - return to neutral
                extension_progress = progress / 0.5
                foot[2] -= self.rear_compression_depth * (1.0 - extension_progress)
                foot[0] -= 0.02 * extension_progress
            else:
                # Liftoff and swing initiation
                swing_progress = (progress - 0.5) / 0.5
                foot[0] += self.vault_step_length * swing_progress * 0.25
                foot[2] += self.rear_swing_height * swing_progress * 0.4
            
        elif phase < 0.75:
            # Phase 0.5-0.75: Continue forward swing with maximum clearance
            # Decouple timing so max height and max forward don't coincide
            progress = smooth_step((phase - 0.5) / 0.25)
            foot[0] += self.vault_step_length * (0.25 + 0.35 * progress)
            # Max height at mid-phase, then descend
            height_phase = np.sin(np.pi * progress)
            foot[2] += self.rear_swing_height * (0.4 + 0.6 * height_phase)
            
        else:
            # Phase 0.75-1.0: Land and settle for next compression with smooth ease-out
            progress = smooth_step((phase - 0.75) / 0.25)
            # Smooth retraction using ease-out curve
            retract_curve = 1.0 - progress * progress
            foot[0] += self.vault_step_length * 0.6 * retract_curve
            # Smooth descent to ground
            foot[2] += self.rear_swing_height * (1.0 - progress) * np.sin(np.pi * (1.0 - progress) * 0.5)
            
        return foot


def smooth_step(t):
    """Smoothstep interpolation for reduced jerk."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)