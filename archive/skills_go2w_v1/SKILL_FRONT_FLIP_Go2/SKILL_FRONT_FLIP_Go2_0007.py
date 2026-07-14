from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_FRONT_FLIP_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Front flip motion for quadruped robot.
    
    Four phases:
    - [0.0, 0.25]: Crouch and pitch preparation
    - [0.25, 0.5]: Extension and launch with forward rotation
    - [0.5, 0.75]: Airborne tuck and continued rotation
    - [0.75, 1.0]: Landing extension and orientation recovery
    
    All leg motions are symmetric to maintain sagittal plane motion.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_pos[2] = 0.10  # Lower initial height for envelope compliance
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters - reduced to prevent joint limit violations
        self.crouch_depth = 0.03
        self.tuck_retraction = 0.035  # Reduced from 0.06
        self.extension_length = 0.03  # Reduced from 0.05
        
        # Base velocity parameters - reduced for altitude envelope compliance
        self.launch_vz = 0.75  # Reduced from 1.4
        self.launch_vx = 0.5
        self.pitch_rate_max = 2.0 * np.pi
        
        # Phase boundaries
        self.phase_crouch_end = 0.25
        self.phase_launch_end = 0.5
        self.phase_tuck_end = 0.75
        self.phase_landing_end = 1.0

    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent velocity commands.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 0: Crouch and pitch prep [0.0, 0.25]
        if phase < self.phase_crouch_end:
            progress = phase / self.phase_crouch_end
            # Smooth crouch with sinusoidal profile
            vz = -0.15 * np.sin(np.pi * progress)
            pitch_rate = 0.3 * np.sin(np.pi * progress)
            vx = 0.05 * progress
        
        # Phase 1: Extension and launch [0.25, 0.5]
        elif phase < self.phase_launch_end:
            progress = (phase - self.phase_crouch_end) / (self.phase_launch_end - self.phase_crouch_end)
            # Smooth launch with controlled peak velocity
            vz = self.launch_vz * np.sin(np.pi * progress)
            vx = self.launch_vx * (0.6 + 0.4 * progress)
            pitch_rate = self.pitch_rate_max * np.sin(np.pi * progress * 0.7)
        
        # Phase 2: Airborne tuck and rotation [0.5, 0.75]
        elif phase < self.phase_tuck_end:
            progress = (phase - self.phase_launch_end) / (self.phase_tuck_end - self.phase_launch_end)
            # Transition to descent immediately, smooth parabolic trajectory
            vx = self.launch_vx * (0.8 - 0.5 * progress)
            vz = -0.9 * progress  # Start descending immediately
            pitch_rate = self.pitch_rate_max * (0.9 + 0.1 * np.cos(np.pi * progress))
        
        # Phase 3: Landing extension and recovery [0.75, 1.0]
        else:
            progress = (phase - self.phase_tuck_end) / (self.phase_landing_end - self.phase_tuck_end)
            # Smooth landing with controlled descent
            vx = self.launch_vx * 0.3 * (1.0 - progress)
            vz = -0.7 * np.sin(np.pi * 0.5 * (1.0 - progress))
            pitch_rate = self.pitch_rate_max * 0.15 * (1.0 - progress)**2
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
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
        Compute foot position in body frame with minimal offsets to respect joint limits.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        is_front = leg_name.startswith('F')
        
        # Reduced base clearance to prevent joint limit violations
        base_clearance = 0.07  # Reduced from 0.12
        
        # Phase 0: Crouch [0.0, 0.25]
        if phase < self.phase_crouch_end:
            progress = phase / self.phase_crouch_end
            # Smooth sinusoidal crouch
            crouch_factor = np.sin(np.pi * 0.5 * progress)
            foot[2] += base_clearance
            foot[2] -= self.crouch_depth * crouch_factor
            if is_front:
                foot[0] -= 0.01 * crouch_factor
            else:
                foot[0] += 0.01 * crouch_factor
        
        # Phase 1: Extension and launch [0.25, 0.5]
        elif phase < self.phase_launch_end:
            progress = (phase - self.phase_crouch_end) / (self.phase_launch_end - self.phase_crouch_end)
            # Smooth transition from crouch to extension
            crouch_factor = 1.0 - progress
            crouch_z = -self.crouch_depth * crouch_factor
            
            extension_z = 0.0
            if progress > 0.4:
                extension_progress = (progress - 0.4) / 0.6
                extension_z = -self.extension_length * np.sin(np.pi * 0.5 * extension_progress)
            
            foot[2] += base_clearance + crouch_z + extension_z
            
            if is_front:
                foot[0] -= 0.01 * (1.0 - progress)
            else:
                foot[0] += 0.01 * (1.0 - progress)
        
        # Phase 2: Airborne tuck [0.5, 0.75]
        elif phase < self.phase_tuck_end:
            progress = (phase - self.phase_launch_end) / (self.phase_tuck_end - self.phase_launch_end)
            # Smooth tuck motion
            extension_factor = 1.0 - np.sin(np.pi * 0.5 * progress)
            extension_z = -self.extension_length * extension_factor
            tuck_z = self.tuck_retraction * np.sin(np.pi * 0.5 * progress)
            
            foot[2] += base_clearance + extension_z + tuck_z
            
            # Narrow stance during tuck
            stance_factor = 1.0 - 0.08 * np.sin(np.pi * 0.5 * progress)
            foot[1] *= stance_factor
        
        # Phase 3: Landing extension [0.75, 1.0]
        else:
            progress = (phase - self.phase_tuck_end) / (self.phase_landing_end - self.phase_tuck_end)
            # Smooth untuck
            tuck_factor = 1.0 - np.sin(np.pi * 0.5 * progress)
            tuck_z = self.tuck_retraction * tuck_factor
            
            # Smooth landing extension
            landing_extension = 0.05 * np.sin(np.pi * 0.5 * progress)
            
            foot[2] += base_clearance + tuck_z - landing_extension
            
            # Return to normal stance
            stance_factor = 0.92 + 0.08 * np.sin(np.pi * 0.5 * progress)
            foot[1] *= stance_factor
        
        return foot