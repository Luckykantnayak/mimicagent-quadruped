from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_REVERSE_SCREW_DRIVE_MotionGenerator(BaseMotionGenerator):
    """
    Reverse screw-drive locomotion: simultaneous backward translation and counter-clockwise yaw rotation.
    
    - All four legs maintain ground contact throughout (sliding/skating motion)
    - Right-side legs (FL, RR) and left-side legs (FR, RL) alternate between active stance (pushing) and repositioning
    - Asymmetric lateral positioning generates yaw torque while backward push generates translation
    - Creates helical trajectory in world frame through integrated velocity
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0  # Hz, full cycle frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Locomotion parameters
        self.backward_velocity = -0.4  # m/s, negative x is backward
        self.yaw_rate = 1.2  # rad/s, positive is counter-clockwise
        
        # Leg motion parameters
        self.stride_length = 0.15  # m, fore-aft foot excursion during stance
        self.lateral_asymmetry = 0.03  # m, lateral offset to generate yaw torque
        
        # Phase offsets for leg groups
        # Right-side legs (FL, RR) push at [0.0-0.25] and [0.5-0.75]
        # Left-side legs (FR, RL) push at [0.25-0.5] and [0.75-1.0]
        self.phase_offsets = {
            leg_names[0]: 0.0,   # FL - right side
            leg_names[1]: 0.25,  # FR - left side
            leg_names[2]: 0.25,  # RL - left side
            leg_names[3]: 0.0,   # RR - right side
        }
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update_base_motion(self, phase, dt):
        """
        Constant backward velocity and counter-clockwise yaw rate throughout all phases.
        """
        # Constant backward velocity in body x direction (world frame via integration)
        self.vel_world = np.array([self.backward_velocity, 0.0, 0.0])
        
        # Constant counter-clockwise yaw rate
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate])
        
        # Integrate pose in world frame
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with sliding stance motion.
        
        Legs alternate between:
        - Active stance (0.25 duration): sweep backward while maintaining contact
        - Repositioning (0.75 duration): slide forward to prepare for next push
        
        All motion keeps foot on ground (z = base_z).
        """
        # Get leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is right-side (FL, RR) or left-side (FR, RL)
        is_right_side = leg_name.startswith('FL') or leg_name.startswith('RR')
        
        # Active stance occurs at [0.0, 0.25] in leg_phase
        # Repositioning occurs at [0.25, 1.0] in leg_phase
        
        if leg_phase < 0.25:
            # Active stance: push backward
            progress = leg_phase / 0.25  # 0 to 1 within stance phase
            
            # Sweep backward: start forward, end rearward
            foot[0] += self.stride_length * (0.5 - progress)
            
            # Apply lateral asymmetry to generate yaw torque
            # Right-side legs push outward, left-side inward (relative to rotation center)
            if is_right_side:
                foot[1] += self.lateral_asymmetry * np.sin(np.pi * progress)
            else:
                foot[1] -= self.lateral_asymmetry * np.sin(np.pi * progress)
                
        else:
            # Repositioning: slide forward to reset for next push
            progress = (leg_phase - 0.25) / 0.75  # 0 to 1 within repositioning phase
            
            # Smooth forward slide from rear to forward position
            # Use cosine for smooth acceleration/deceleration
            slide_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            # Position: start at rear (-0.5 * stride), end at forward (+0.5 * stride)
            foot[0] += self.stride_length * (-0.5 + slide_progress)
            
            # Gradually reduce lateral offset during repositioning
            lateral_factor = 1.0 - progress
            if is_right_side:
                foot[1] += self.lateral_asymmetry * lateral_factor * 0.3
            else:
                foot[1] -= self.lateral_asymmetry * lateral_factor * 0.3
        
        # Z remains at ground level (no lifting)
        #foot[2] unchanged from base position
        
        return foot