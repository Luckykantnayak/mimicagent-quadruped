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
        self.vertical_compression = 0.08  # Vertical compression amount
        
        # Base velocity parameters
        self.surge_vx_max = 1.5  # Peak forward velocity during surge
        self.surge_vz_up = 0.6  # Upward velocity during surge
        self.surge_vz_down = -0.4  # Downward velocity during compression
        self.glide_decay_rate = 0.7  # Velocity decay during glide
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # All legs move synchronously (no phase offsets)
        self.phase_offsets = {leg: 0.0 for leg in leg_names}

    def update_base_motion(self, phase, dt):
        """
        Update base velocity according to pulse-surge-glide cycle.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        # Phase 0.0-0.2: Compression gather
        if phase < 0.2:
            local_phase = phase / 0.2
            vx = -0.2 * np.sin(np.pi * local_phase)  # Slight backward drift
            vz = self.surge_vz_down * (1.0 - np.cos(np.pi * local_phase)) / 2.0  # Downward
            
        # Phase 0.2-0.4: Explosive surge
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Rapid acceleration in forward direction
            vx = self.surge_vx_max * np.sin(np.pi * local_phase)
            # Upward lift during surge
            vz = self.surge_vz_up * np.sin(np.pi * local_phase)
            
        # Phase 0.4-0.6: Extended glide
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Decay forward velocity during glide
            vx = self.surge_vx_max * self.glide_decay_rate * (1.0 - local_phase)
            # Slight downward drift
            vz = -0.1 * local_phase
            
        # Phase 0.6-0.8: Gather transition
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Continued velocity decay
            vx = self.surge_vx_max * self.glide_decay_rate * (1.0 - local_phase) * 0.3
            # Gradual descent
            vz = -0.2 * local_phase
            
        # Phase 0.8-1.0: Reset compression
        else:
            local_phase = (phase - 0.8) / 0.2
            # Minimal backward drift
            vx = -0.15 * np.sin(np.pi * local_phase)
            # Downward to complete compression
            vz = self.surge_vz_down * (1.0 - np.cos(np.pi * local_phase)) / 2.0
        
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

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for synchronized compression-extension cycle.
        All legs move together through gather-extend-glide-gather pattern.
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
            gather_progress = local_phase
            
            # Gather inward (medial motion)
            lateral_compression = 1.0 - (1.0 - self.compression_factor) * gather_progress
            foot[1] = base_pos[1] * lateral_compression
            
            # Move toward body longitudinally
            longitudinal_gather = 0.5 - 0.5 * np.cos(np.pi * gather_progress)
            if is_front:
                foot[0] = base_pos[0] - 0.06 * longitudinal_gather
            else:
                foot[0] = base_pos[0] + 0.06 * longitudinal_gather
            
            # Vertical compression (foot stays on ground, effective leg shortening)
            foot[2] = base_pos[2] + self.vertical_compression * gather_progress
            
        # Phase 0.2-0.4: Explosive surge extension
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            extension_progress = local_phase
            
            # Extend from compressed to extended position
            # Lateral extension
            lateral_expansion = self.compression_factor + (1.0 - self.compression_factor) * extension_progress
            foot[1] = base_pos[1] * lateral_expansion
            
            # Longitudinal extension with forward reach
            longitudinal_factor = 0.5 - 0.5 * np.cos(np.pi * (1.0 - extension_progress))
            if is_front:
                foot[0] = base_pos[0] - 0.06 * longitudinal_factor + self.extension_reach * extension_progress
            else:
                foot[0] = base_pos[0] + 0.06 * longitudinal_factor - self.extension_reach * extension_progress
            
            # Vertical extension (return to nominal height and push down)
            foot[2] = base_pos[2] + self.vertical_compression * (1.0 - extension_progress)
            
        # Phase 0.4-0.6: Extended glide (hold position)
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            
            # Maintain extended posture with minimal variation
            foot[1] = base_pos[1]
            
            if is_front:
                foot[0] = base_pos[0] + self.extension_reach
            else:
                foot[0] = base_pos[0] - self.extension_reach
            
            foot[2] = base_pos[2]
            
        # Phase 0.6-0.8: Gather transition begins
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            gather_progress = local_phase * 0.5  # Partial gathering
            
            # Begin medial motion
            lateral_compression = 1.0 - (1.0 - self.compression_factor) * gather_progress
            foot[1] = base_pos[1] * lateral_compression
            
            # Begin longitudinal gathering
            if is_front:
                foot[0] = base_pos[0] + self.extension_reach * (1.0 - gather_progress)
            else:
                foot[0] = base_pos[0] - self.extension_reach * (1.0 - gather_progress)
            
            # Slight vertical compression begins
            foot[2] = base_pos[2] + self.vertical_compression * gather_progress * 0.3
            
        # Phase 0.8-1.0: Reset compression (complete gathering)
        else:
            local_phase = (phase - 0.8) / 0.2
            # Map from partial gathering (0.5) to full compression (1.0)
            gather_progress = 0.5 + 0.5 * local_phase
            
            # Complete medial gathering
            lateral_compression = 1.0 - (1.0 - self.compression_factor) * gather_progress
            foot[1] = base_pos[1] * lateral_compression
            
            # Complete longitudinal gathering
            longitudinal_factor = 0.5 - 0.5 * np.cos(np.pi * (gather_progress - 0.5) / 0.5)
            if is_front:
                foot[0] = base_pos[0] - 0.06 * longitudinal_factor
            else:
                foot[0] = base_pos[0] + 0.06 * longitudinal_factor
            
            # Complete vertical compression
            foot[2] = base_pos[2] + self.vertical_compression * gather_progress
        
        return foot