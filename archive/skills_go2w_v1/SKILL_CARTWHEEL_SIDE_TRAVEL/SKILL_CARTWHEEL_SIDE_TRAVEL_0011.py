from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_CARTWHEEL_SIDE_TRAVEL_MotionGenerator(BaseMotionGenerator):
    """
    Cartwheel motion with full 360-degree roll rotation and lateral displacement.
    
    The robot performs a dynamic cartwheel by:
    - Rolling continuously around the x-axis (360 degrees per cycle)
    - Moving laterally (positive y direction) 
    - Sequencing leg contacts with world-frame-aware foot positioning
    - Legs trace arcs that remain kinematically reachable during all roll angles
    - Base height maintained within safe envelope with adequate inversion clearance
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # 2 seconds per cycle
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Extract nominal offsets for world-frame positioning
        self.nominal_offsets = {}
        for leg in leg_names:
            self.nominal_offsets[leg] = initial_foot_positions_body[leg].copy()
        
        # Cartwheel parameters
        self.roll_rate_max = 2.0 * np.pi * self.freq  # 360 degrees per cycle
        self.lateral_velocity = 0.16  # Controlled rightward velocity
        self.forward_velocity = 0.04  # Slight forward momentum
        
        # Conservative arc parameters
        self.swing_clearance = 0.14  # Max height above ground during swing (world frame)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity tracking
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base motion with continuous roll rotation and enhanced vertical clearance.
        Higher base height during inversion (0.56-0.62m) provides workspace for legs.
        """
        
        # Roll rate profile: smooth acceleration and deceleration
        if phase < 0.25:
            roll_progress = phase / 0.25
            roll_rate = self.roll_rate_max * np.sin(np.pi * roll_progress / 2)
        elif phase < 0.75:
            roll_rate = self.roll_rate_max
        else:
            roll_progress = (phase - 0.75) / 0.25
            roll_rate = self.roll_rate_max * np.cos(np.pi * roll_progress / 2)
        
        # Lateral velocity profile
        if phase < 0.1:
            vy = self.lateral_velocity * (phase / 0.1)
        elif phase < 0.85:
            vy = self.lateral_velocity
        else:
            vy = self.lateral_velocity * (1.0 - (phase - 0.85) / 0.15)
        
        vx = self.forward_velocity
        
        # Enhanced vertical velocity with increased bias during inversion
        if phase < 0.2:
            # Initial ascent
            vz = 0.45 * np.sin(np.pi * phase / 0.2)
        elif phase < 0.6:
            # Maintain higher elevation during inversion with stronger upward bias
            base_vz = 0.35 * np.sin(2 * np.pi * phase)
            vz = base_vz + 0.15  # Increased bias for 0.56-0.62m peak height
        elif phase < 0.85:
            # Controlled descent
            vz = 0.3 * np.sin(2 * np.pi * phase)
        else:
            # Final damping
            damping_progress = (phase - 0.85) / 0.15
            vz = 0.3 * np.sin(2 * np.pi * phase) * (1.0 - damping_progress) - 0.15 * damping_progress
        
        # Soft height limiting
        current_height = self.root_pos[2]
        if current_height > 0.62:
            vz = min(vz, -0.2)
        elif current_height < 0.28:
            vz = max(vz, 0.2)
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def world_to_body_frame(self, world_pos):
        """Transform world frame position to body frame using inverse quaternion."""
        relative_pos = world_pos - self.root_pos
        quat_inv = quaternion_inverse(self.root_quat)
        body_pos = quaternion_rotate(quat_inv, relative_pos)
        return body_pos

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position using world-frame positioning during swing phases.
        This ensures reachability regardless of base roll angle.
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        is_front_leg = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Nominal offsets for world-frame positioning
        fore_aft_offset = 0.20 if is_front_leg else -0.20
        lateral_offset = 0.14 if is_right_leg else -0.14
        
        if is_right_leg:
            # Right legs: stance [0, 0.27], swing [0.27, 0.73], return [0.73, 1.0]
            
            if phase < 0.27:
                # Stance phase: body-frame positioning
                foot = base_pos.copy()
                foot[2] += 0.005 * np.sin(np.pi * phase / 0.27)
                return foot
            
            elif phase < 0.73:
                # Swing phase: world-frame clearance with reduced arc (75 degrees)
                swing_progress = (phase - 0.27) / 0.46
                
                # Sinusoidal clearance profile (peaks at mid-swing)
                clearance = self.swing_clearance * np.sin(np.pi * swing_progress)
                
                # World frame target position
                world_foot = np.array([
                    self.root_pos[0] + fore_aft_offset * 0.9,  # Slightly reduced spread
                    self.root_pos[1] + lateral_offset,
                    clearance  # Above ground
                ])
                
                # Transform to body frame
                foot = self.world_to_body_frame(world_foot)
                return foot
            
            else:
                # Return phase: smooth transition to stance
                return_progress = (phase - 0.73) / 0.27
                
                # Blend from small clearance to ground contact
                if return_progress < 0.5:
                    # First half: descend to near-ground
                    clearance = 0.04 * (1.0 - 2.0 * return_progress)
                    world_foot = np.array([
                        self.root_pos[0] + fore_aft_offset * 0.9,
                        self.root_pos[1] + lateral_offset,
                        clearance
                    ])
                    foot = self.world_to_body_frame(world_foot)
                else:
                    # Second half: settle to body-frame stance
                    blend_weight = (return_progress - 0.5) / 0.5
                    world_foot = np.array([
                        self.root_pos[0] + fore_aft_offset * 0.9,
                        self.root_pos[1] + lateral_offset,
                        0.01
                    ])
                    world_based = self.world_to_body_frame(world_foot)
                    body_based = base_pos.copy()
                    foot = (1.0 - blend_weight) * world_based + blend_weight * body_based
                
                return foot
        
        else:
            # Left legs: swing [0, 0.27], transition [0.27, 0.6], stance [0.6, 0.82], return [0.82, 1.0]
            
            if phase < 0.27:
                # Early swing: world-frame positioning
                swing_progress = phase / 0.27
                clearance = self.swing_clearance * np.sin(np.pi * swing_progress * 0.7)
                
                world_foot = np.array([
                    self.root_pos[0] + fore_aft_offset * 0.9,
                    self.root_pos[1] + lateral_offset,
                    clearance
                ])
                
                foot = self.world_to_body_frame(world_foot)
                return foot
            
            elif phase < 0.6:
                # Transition swing: descending to contact
                swing_progress = (phase - 0.27) / 0.33
                clearance = self.swing_clearance * 0.5 * np.sin(np.pi * (0.7 + swing_progress * 0.3))
                
                # Reduce clearance as approaching stance
                clearance *= (1.0 - swing_progress * 0.7)
                
                world_foot = np.array([
                    self.root_pos[0] + fore_aft_offset * 0.9,
                    self.root_pos[1] + lateral_offset,
                    max(clearance, 0.01)
                ])
                
                foot = self.world_to_body_frame(world_foot)
                return foot
            
            elif phase < 0.82:
                # Stance phase during inversion: direct ground projection
                # Target ground contact in world frame, transform to body
                world_foot = np.array([
                    self.root_pos[0] + fore_aft_offset * 0.85,
                    self.root_pos[1] + lateral_offset * 0.95,
                    0.015  # Slight elevation for contact
                ])
                
                foot = self.world_to_body_frame(world_foot)
                return foot
            
            else:
                # Return to nominal stance
                return_progress = (phase - 0.82) / 0.18
                
                # Blend from ground contact to body-frame stance
                world_foot = np.array([
                    self.root_pos[0] + fore_aft_offset * 0.85,
                    self.root_pos[1] + lateral_offset * 0.95,
                    0.01
                ])
                world_based = self.world_to_body_frame(world_foot)
                body_based = base_pos.copy()
                
                foot = (1.0 - return_progress) * world_based + return_progress * body_based
                return foot