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
        self.freq = 0.8  # Cycle frequency for vaulting rhythm
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.rear_compression_depth = 0.12  # Vertical compression during loading
        self.front_extension_length = 0.15  # Forward reach of front legs
        self.rear_swing_height = 0.18  # Ground clearance during rear leg swing
        self.front_swing_height = 0.08  # Front leg clearance during flight
        self.vault_step_length = 0.25  # Forward displacement per cycle
        
        # Base velocity parameters
        self.thrust_vx_peak = 1.2  # Peak forward velocity during thrust
        self.thrust_vz_peak = 0.6  # Peak upward velocity during thrust
        self.landing_vx = 0.7  # Forward velocity during landing
        self.landing_vz = -0.4  # Downward velocity during landing
        
        # Pitch control parameters
        self.pitch_rate_back = -1.5  # Negative pitch rate during compression
        self.pitch_rate_forward = 2.5  # Positive pitch rate during thrust and landing
        self.pitch_rate_stabilize = -1.2  # Negative pitch rate during transition
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base velocity and angular velocity based on phase.
        """
        vx = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        if phase < 0.25:
            # Phase 0.0-0.25: Rear compression, backward pitch preparation
            progress = phase / 0.25
            vx = 0.1 * progress
            vz = -0.05 * np.sin(np.pi * progress)
            pitch_rate = self.pitch_rate_back * np.sin(np.pi * progress)
            
        elif phase < 0.5:
            # Phase 0.25-0.5: Explosive thrust and flight
            progress = (phase - 0.25) / 0.25
            vx = self.thrust_vx_peak * np.sin(np.pi * progress * 0.5 + np.pi * 0.5)
            vz = self.thrust_vz_peak * np.sin(np.pi * progress)
            pitch_rate = self.pitch_rate_forward * (1.0 - 0.3 * progress)
            
        elif phase < 0.75:
            # Phase 0.5-0.75: Front landing, continued forward pitch
            progress = (phase - 0.5) / 0.25
            vx = self.landing_vx * (1.0 - 0.4 * progress)
            vz = self.landing_vz * np.sin(np.pi * progress * 0.5)
            pitch_rate = self.pitch_rate_forward * (0.8 - 0.6 * progress)
            
        else:
            # Phase 0.75-1.0: Rear touchdown and transition
            progress = (phase - 0.75) / 0.25
            vx = 0.3 * (1.0 - progress)
            vz = 0.0
            pitch_rate = self.pitch_rate_stabilize * progress
        
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
        """
        foot = foot_base.copy()
        
        if phase < 0.25:
            # Phase 0.0-0.25: Extend forward during rear compression
            progress = phase / 0.25
            foot[0] += self.front_extension_length * progress
            foot[2] += 0.02 * progress
            
        elif phase < 0.5:
            # Phase 0.25-0.5: Lift off during flight
            progress = (phase - 0.25) / 0.25
            foot[0] += self.front_extension_length * (1.0 - 0.3 * progress)
            foot[2] += self.front_swing_height * np.sin(np.pi * progress)
            
        elif phase < 0.75:
            # Phase 0.5-0.75: Land and sweep backward in body frame
            progress = (phase - 0.5) / 0.25
            foot[0] += self.front_extension_length * (0.7 - 0.7 * progress)
            foot[2] -= 0.01 * progress  # Slight compression on landing
            
        else:
            # Phase 0.75-1.0: Transition to next cycle
            progress = (phase - 0.75) / 0.25
            foot[0] += self.front_extension_length * progress
            
        return foot
    
    def _compute_rear_leg_trajectory(self, foot_base, phase):
        """
        Rear leg trajectory through vault cycle.
        """
        foot = foot_base.copy()
        
        if phase < 0.25:
            # Phase 0.0-0.25: Compress for thrust loading
            progress = phase / 0.25
            foot[2] -= self.rear_compression_depth * np.sin(np.pi * progress * 0.5)
            foot[0] += 0.03 * progress  # Slight forward shift during compression
            
        elif phase < 0.5:
            # Phase 0.25-0.5: Explosive extension and liftoff, begin forward swing
            progress = (phase - 0.25) / 0.25
            if progress < 0.4:
                # Extension phase
                extension_progress = progress / 0.4
                foot[2] -= self.rear_compression_depth * (1.0 - extension_progress)
                foot[0] -= 0.05 * extension_progress
            else:
                # Liftoff and swing initiation
                swing_progress = (progress - 0.4) / 0.6
                foot[0] += self.vault_step_length * swing_progress * 0.3
                foot[2] += self.rear_swing_height * swing_progress * 0.5
            
        elif phase < 0.75:
            # Phase 0.5-0.75: Continue forward swing with maximum clearance
            progress = (phase - 0.5) / 0.25
            foot[0] += self.vault_step_length * (0.3 + 0.5 * progress)
            foot[2] += self.rear_swing_height * (0.5 + 0.5 * np.sin(np.pi * progress))
            
        else:
            # Phase 0.75-1.0: Land and settle for next compression
            progress = (phase - 0.75) / 0.25
            foot[0] += self.vault_step_length * (0.8 - 0.8 * progress)
            foot[2] += self.rear_swing_height * (1.0 - np.sin(np.pi * progress * 0.5))
            
        return foot