from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_PULSE_SURGE_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Pulse Surge Forward: Rhythmic forward locomotion with compression-extension-glide cycles.
    
    Motion phases:
    - [0.0-0.2]: Compression gather - legs compress and gather beneath body
    - [0.2-0.4]: Explosive surge - rapid leg extension drives forward and upward
    - [0.4-0.6]: Extended glide - coast forward with momentum in extended posture
    - [0.6-0.8]: Gather transition - legs begin gathering inward while coasting
    - [0.8-1.0]: Reset compression - complete compression to match starting state
    
    All four legs move synchronously throughout the cycle.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency for rhythmic pulsing
        
        # Base foot positions (extended stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters for leg trajectories
        self.compression_factor = 0.6  # How much legs gather inward (0.6 = 60% of original stance width)
        self.extension_reach = 0.12  # Additional forward/rearward reach during extension
        self.vertical_extension = 0.18  # Increased from 0.08 to provide sufficient downward reach in world frame
        
        # Base velocity parameters - balanced vertical motion to prevent net drift
        self.surge_vx_max = 1.5  # Peak forward velocity during surge
        self.surge_vz_up = 0.24  # Increased upward velocity to balance downward phases
        self.surge_vz_down = -0.16  # Reduced downward velocity to limit descent rate
        self.glide_decay_rate = 0.7  # Velocity decay during glide
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # All legs move synchronously (no phase offsets)
        self.phase_offsets = {leg: 0.0 for leg in leg_names}
        
        # Track integrated base vertical displacement for foot-base coupling
        self.base_z_offset = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base velocity according to pulse-surge-glide cycle.
        Vertical velocity is balanced to prevent net drift and maintain ground contact.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        # Phase 0.0-0.2: Compression gather
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth transition into phase
            smoothing = 0.5 - 0.5 * np.cos(np.pi * local_phase)
            vx = -0.2 * np.sin(np.pi * local_phase)  # Slight backward drift
            # Downward motion during compression, reduced magnitude
            vz = self.surge_vz_down * smoothing
            
        # Phase 0.2-0.4: Explosive surge
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Smooth ramp up and down
            surge_envelope = np.sin(np.pi * local_phase)
            # Rapid acceleration in forward direction
            vx = self.surge_vx_max * surge_envelope
            # Increased upward lift during surge to balance cycle
            vz = self.surge_vz_up * surge_envelope
            
        # Phase 0.4-0.6: Extended glide with slight upward lift to maintain height
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Smooth decay
            decay_factor = np.cos(np.pi * local_phase / 2.0)
            # Decay forward velocity during glide
            vx = self.surge_vx_max * self.glide_decay_rate * decay_factor
            # Slight upward motion to maintain base height instead of descending
            vz = 0.02 * (1.0 - local_phase)
            
        # Phase 0.6-0.8: Gather transition with minimal vertical change
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Smooth continued decay
            decay_factor = np.cos(np.pi * (0.5 + local_phase / 2.0))
            # Continued velocity decay
            vx = self.surge_vx_max * self.glide_decay_rate * 0.3 * decay_factor
            # Minimal vertical motion, transitioning to descent
            vz = -0.02 * np.sin(np.pi * local_phase / 2.0)
            
        # Phase 0.8-1.0: Reset compression
        else:
            local_phase = (phase - 0.8) / 0.2
            # Smooth blending into next cycle
            smoothing = 0.5 - 0.5 * np.cos(np.pi * local_phase)
            # Minimal backward drift
            vx = -0.15 * np.sin(np.pi * local_phase)
            # Downward to complete compression, smooth into cycle restart
            vz = self.surge_vz_down * smoothing
        
        # Set velocities in world frame
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.zeros(3)  # No rotation during pulse motion
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Track base vertical displacement for enhanced foot-base coupling
        self.base_z_offset += vz * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for synchronized compression-extension cycle.
        All legs move together through gather-extend-glide-gather pattern.
        
        Key principle: When base descends (compression), feet extend DOWN in world frame
        (appear to move UP in body frame) to maintain ground contact.
        Enhanced coupling ensures feet extend sufficiently to compensate for base descent.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine if front or rear leg for longitudinal motion direction
        is_front = leg_name.startswith('F')
        # Determine lateral side for medial gathering
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Phase 0.0-0.2: Compression gather
        if phase < 0.2:
            local_phase = phase / 0.2
            gather_progress = 0.5 - 0.5 * np.cos(np.pi * local_phase)
            
            # Gather inward (medial motion)
            lateral_compression = 1.0 - (1.0 - self.compression_factor) * gather_progress
            foot[1] = base_pos[1] * lateral_compression
            
            # Move toward body longitudinally
            if is_front:
                foot[0] = base_pos[0] - 0.06 * gather_progress
            else:
                foot[0] = base_pos[0] + 0.06 * gather_progress
            
            # Vertical: feet extend DOWN in world (UP in body) during compression
            # Increased extension magnitude to maintain contact as base descends
            foot[2] = base_pos[2] + self.vertical_extension * gather_progress
            
        # Phase 0.2-0.4: Explosive surge extension
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            extension_progress = 0.5 - 0.5 * np.cos(np.pi * local_phase)
            
            # Extend from compressed to extended position
            # Lateral extension
            lateral_expansion = self.compression_factor + (1.0 - self.compression_factor) * extension_progress
            foot[1] = base_pos[1] * lateral_expansion
            
            # Longitudinal extension with forward reach
            if is_front:
                foot[0] = base_pos[0] - 0.06 * (1.0 - extension_progress) + self.extension_reach * extension_progress
            else:
                foot[0] = base_pos[0] + 0.06 * (1.0 - extension_progress) - self.extension_reach * extension_progress
            
            # Vertical: return from extended position to nominal as base rises
            foot[2] = base_pos[2] + self.vertical_extension * (1.0 - extension_progress)
            
        # Phase 0.4-0.6: Extended glide
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            
            # Maintain extended posture
            foot[1] = base_pos[1]
            
            if is_front:
                foot[0] = base_pos[0] + self.extension_reach
            else:
                foot[0] = base_pos[0] - self.extension_reach
            
            # Maintain nominal height with minimal adjustment
            foot[2] = base_pos[2] - 0.01 * local_phase
            
        # Phase 0.6-0.8: Gather transition begins
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            gather_progress = 0.5 * (0.5 - 0.5 * np.cos(np.pi * local_phase))
            
            # Begin medial motion
            lateral_compression = 1.0 - (1.0 - self.compression_factor) * gather_progress
            foot[1] = base_pos[1] * lateral_compression
            
            # Begin longitudinal gathering
            reach_factor = 1.0 - gather_progress / 0.5
            if is_front:
                foot[0] = base_pos[0] + self.extension_reach * reach_factor
            else:
                foot[0] = base_pos[0] - self.extension_reach * reach_factor
            
            # Feet begin extending down in world frame (up in body frame) early
            foot[2] = base_pos[2] - 0.01 + (self.vertical_extension + 0.01) * gather_progress
            
        # Phase 0.8-1.0: Reset compression (complete gathering)
        else:
            local_phase = (phase - 0.8) / 0.2
            # Map from partial gathering (0.5) to full compression (1.0)
            partial_progress = 0.5 - 0.5 * np.cos(np.pi * local_phase)
            gather_progress = 0.5 + 0.5 * partial_progress
            
            # Complete medial gathering
            lateral_compression = 1.0 - (1.0 - self.compression_factor) * gather_progress
            foot[1] = base_pos[1] * lateral_compression
            
            # Complete longitudinal gathering
            if is_front:
                foot[0] = base_pos[0] - 0.06 * gather_progress
            else:
                foot[0] = base_pos[0] + 0.06 * gather_progress
            
            # Complete vertical extension in body frame (feet reach down in world frame)
            foot[2] = base_pos[2] + self.vertical_extension * gather_progress
        
        return foot