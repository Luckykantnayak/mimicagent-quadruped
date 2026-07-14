from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_CARTWHEEL_SIDE_TRAVEL_MotionGenerator(BaseMotionGenerator):
    """
    Cartwheel motion with full 360-degree roll rotation and lateral displacement.
    
    The robot performs a dynamic cartwheel by:
    - Rolling continuously around the x-axis (360 degrees per cycle)
    - Moving laterally (positive y direction) 
    - Sequencing leg contacts: right legs plant [0, 0.3], aerial [0.3, 0.6],
      left legs plant [0.6, 0.85], all legs return [0.85, 1.0]
    - Legs trace constrained circular arcs with continuous phase transitions
    - Base height maintained within safe envelope with adequate clearance during inversion
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # 2 seconds per cycle
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Cartwheel parameters
        self.roll_rate_max = 2.0 * np.pi * self.freq  # 360 degrees per cycle
        self.lateral_velocity = 0.18  # Controlled rightward velocity
        self.forward_velocity = 0.04  # Slight forward momentum
        
        # Conservative leg arc parameters
        self.arc_radius = 0.16  # Reduced for knee safety
        self.arc_height_offset = 0.02  # Minimal vertical offset
        
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
        Maintains base height between 0.48-0.62m during inversion for leg workspace.
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
        
        # Enhanced vertical velocity with higher clearance during inversion
        if phase < 0.2:
            # Initial ascent
            vz = 0.4 * np.sin(np.pi * phase / 0.2)
        elif phase < 0.6:
            # Maintain elevation during inversion with upward bias
            base_vz = 0.35 * np.sin(2 * np.pi * phase)
            vz = base_vz + 0.08  # Upward bias to maintain 0.50-0.60m height
        elif phase < 0.85:
            # Controlled descent during recovery
            vz = 0.3 * np.sin(2 * np.pi * phase)
        else:
            # Final damping to return to nominal
            damping_progress = (phase - 0.85) / 0.15
            vz = 0.3 * np.sin(2 * np.pi * phase) * (1.0 - damping_progress) - 0.1 * damping_progress
        
        # Soft height limiting
        current_height = self.root_pos[2]
        if current_height > 0.60:
            vz = min(vz, -0.15)
        elif current_height < 0.28:
            vz = max(vz, 0.15)
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position with phase-continuous trajectories and roll-aware positioning.
        Critical fixes: inverted stance uses flipped Z logic, phase boundaries use blending.
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Approximate current roll angle
        roll_angle = 2.0 * np.pi * phase
        
        # Adaptive arc scaling
        if phase > 0.25 and phase < 0.65:
            arc_scale = 0.8
        else:
            arc_scale = 1.0
        
        effective_radius = self.arc_radius * arc_scale
        
        if is_right_leg:
            # Right legs: extended contact phase [0, 0.3], swing [0.3, 0.7], return [0.7, 1.0]
            
            if phase < 0.3:
                # Stance phase: minimal motion
                foot = base_pos.copy()
                foot[2] += 0.005 * np.sin(np.pi * phase / 0.3)
                return foot
            
            elif phase < 0.7:
                # Swing phase: 90-degree arc for conservative clearance
                swing_progress = (phase - 0.3) / 0.4
                arc_angle = swing_progress * (np.pi / 2)  # 90 degrees
                
                foot = base_pos.copy()
                foot[1] += effective_radius * np.sin(arc_angle) * 0.15
                foot[2] += effective_radius * (1.0 - np.cos(arc_angle))
                
                # Roll-aware adjustment during inversion
                if roll_angle > 0.7 * np.pi and roll_angle < 1.3 * np.pi:
                    # Reduce Z extension when inverted
                    foot[2] *= 0.65
                
                return foot
            
            else:
                # Return phase with blending zone [0.7, 1.0]
                return_progress = (phase - 0.7) / 0.3
                
                # Compute swing end position for blending
                if phase < 0.72:
                    # Still blending from swing
                    blend_weight = (phase - 0.7) / 0.02
                    swing_foot = base_pos.copy()
                    swing_foot[2] += effective_radius * 0.65
                    
                    return_foot = base_pos.copy()
                    return_foot[2] += 0.01
                    
                    foot = (1.0 - blend_weight) * swing_foot + blend_weight * return_foot
                    return foot
                
                # Smooth descent to nominal
                foot = base_pos.copy()
                descent_height = 0.03 * (1.0 - min(return_progress / 0.7, 1.0))
                foot[2] += descent_height
                
                return foot
        
        else:
            # Left legs: swing [0, 0.3], transition [0.3, 0.6], stance [0.6, 0.85], return [0.85, 1.0]
            
            if phase < 0.3:
                # Early swing: reduced arc
                swing_progress = phase / 0.3
                arc_angle = swing_progress * 0.45 * np.pi  # 81 degrees
                
                foot = base_pos.copy()
                foot[1] -= effective_radius * np.sin(arc_angle) * 0.15
                foot[2] += effective_radius * (1.0 - np.cos(arc_angle)) + self.arc_height_offset
                
                return foot
            
            elif phase < 0.6:
                # Transition swing toward contact
                swing_progress = (phase - 0.3) / 0.3
                arc_angle = 0.45 * np.pi + swing_progress * 0.35 * np.pi  # 81 to 144 degrees
                
                foot = base_pos.copy()
                foot[1] -= effective_radius * np.sin(arc_angle) * 0.15
                z_height = effective_radius * (1.0 - np.cos(arc_angle)) + self.arc_height_offset
                # Smooth descent toward contact
                z_height *= (1.0 - swing_progress * 0.6)
                foot[2] += z_height
                
                return foot
            
            elif phase < 0.85:
                # Inverted stance phase: CORRECTED Z LOGIC
                stance_progress = (phase - 0.6) / 0.25
                foot = base_pos.copy()
                
                # When inverted (roll ~180-270°), flip Z coordinate to reach ground
                if roll_angle > 2.4 and roll_angle < 4.8:  # ~137-275 degrees
                    # Body is inverted: foot needs positive Z in body frame to reach ground
                    foot[2] = -base_pos[2] * 0.6 + 0.03
                else:
                    # Transitioning to/from inversion
                    foot[2] = base_pos[2] * 0.5
                
                return foot
            
            else:
                # Return to nominal
                return_progress = (phase - 0.85) / 0.15
                foot = base_pos.copy()
                
                # Smooth transition from inverted stance to upright stance
                if roll_angle > 4.7:  # Near completion of rotation
                    z_offset = 0.015 * (1.0 - return_progress)
                    foot[2] = base_pos[2] + z_offset
                else:
                    foot[2] = base_pos[2] * (0.5 + 0.5 * return_progress)
                
                return foot