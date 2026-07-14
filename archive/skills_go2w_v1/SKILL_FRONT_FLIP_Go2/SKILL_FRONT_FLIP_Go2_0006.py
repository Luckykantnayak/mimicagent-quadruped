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
        self.root_pos[2] = 0.15  # Elevated initial base height for clearance
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.crouch_depth = 0.03
        self.tuck_retraction = 0.06
        self.extension_length = 0.05
        
        # Base velocity parameters - increased for adequate altitude
        self.launch_vz = 1.4
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
            vz = -0.15 * np.sin(np.pi * progress)
            pitch_rate = 0.3 * np.sin(np.pi * progress)
            vx = 0.05 * progress
        
        # Phase 1: Extension and launch [0.25, 0.5]
        elif phase < self.phase_launch_end:
            progress = (phase - self.phase_crouch_end) / (self.phase_launch_end - self.phase_crouch_end)
            vz = self.launch_vz * np.sin(np.pi * progress * 0.8)
            vx = self.launch_vx * (0.6 + 0.4 * progress)
            pitch_rate = self.pitch_rate_max * np.sin(np.pi * progress * 0.7)
        
        # Phase 2: Airborne tuck and rotation [0.5, 0.75]
        elif phase < self.phase_tuck_end:
            progress = (phase - self.phase_launch_end) / (self.phase_tuck_end - self.phase_launch_end)
            vx = self.launch_vx * (0.8 - 0.5 * progress)
            vz = self.launch_vz * 0.3 * (1.0 - progress) - 0.8 * progress
            pitch_rate = self.pitch_rate_max * (1.0 - 0.1 * progress)
        
        # Phase 3: Landing extension and recovery [0.75, 1.0]
        else:
            progress = (phase - self.phase_tuck_end) / (self.phase_landing_end - self.phase_tuck_end)
            vx = self.launch_vx * 0.3 * (1.0 - progress)
            vz = -0.6 * progress
            pitch_rate = self.pitch_rate_max * 0.2 * (1.0 - progress)**2
        
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
        Compute foot position in body frame with pitch-compensated clearance.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        is_front = leg_name.startswith('F')
        
        # Estimate pitch angle from phase for clearance compensation
        pitch_angle = self._estimate_pitch_angle(phase)
        
        # Base clearance offset - larger to prevent initial penetration
        base_clearance = 0.12
        
        # Phase 0: Crouch [0.0, 0.25]
        if phase < self.phase_crouch_end:
            progress = phase / self.phase_crouch_end
            foot[2] += base_clearance
            foot[2] -= self.crouch_depth * progress
            if is_front:
                foot[0] -= 0.01 * progress
            else:
                foot[0] += 0.01 * progress
        
        # Phase 1: Extension and launch [0.25, 0.5]
        elif phase < self.phase_launch_end:
            progress = (phase - self.phase_crouch_end) / (self.phase_launch_end - self.phase_crouch_end)
            crouch_z = -self.crouch_depth * (1.0 - progress)
            extension_z = 0.0
            if progress > 0.4:
                extension_progress = (progress - 0.4) / 0.6
                extension_z = -self.extension_length * extension_progress
            
            foot[2] += base_clearance + crouch_z + extension_z
            
            # Pitch compensation: as body pitches forward, add z-offset
            pitch_compensation = abs(pitch_angle) * 0.08
            foot[2] += pitch_compensation
            
            if is_front:
                foot[0] -= 0.01 * (1.0 - progress)
            else:
                foot[0] += 0.01 * (1.0 - progress)
        
        # Phase 2: Airborne tuck [0.5, 0.75]
        elif phase < self.phase_tuck_end:
            progress = (phase - self.phase_launch_end) / (self.phase_tuck_end - self.phase_launch_end)
            extension_z = -self.extension_length * (1.0 - progress)
            tuck_z = self.tuck_retraction * progress
            
            foot[2] += base_clearance + extension_z + tuck_z
            
            # Maximum pitch compensation during inverted phase
            pitch_compensation = abs(pitch_angle) * 0.12
            foot[2] += pitch_compensation
            
            foot[1] *= (1.0 - 0.08 * progress)
        
        # Phase 3: Landing extension [0.75, 1.0]
        else:
            progress = (phase - self.phase_tuck_end) / (self.phase_landing_end - self.phase_tuck_end)
            tuck_z = self.tuck_retraction * (1.0 - progress)
            
            # Gentler landing extension, coordinated with base descent
            landing_extension = 0.06 * progress
            
            foot[2] += base_clearance + tuck_z - landing_extension
            
            # Reduce pitch compensation as body returns upright
            pitch_compensation = abs(pitch_angle) * 0.08 * (1.0 - progress)
            foot[2] += pitch_compensation
            
            foot[1] *= (0.92 + 0.08 * progress)
        
        return foot
    
    def _estimate_pitch_angle(self, phase):
        """
        Estimate pitch angle from phase for clearance compensation.
        Returns approximate pitch in radians.
        """
        if phase < self.phase_crouch_end:
            progress = phase / self.phase_crouch_end
            return 0.1 * progress
        elif phase < self.phase_launch_end:
            progress = (phase - self.phase_crouch_end) / (self.phase_launch_end - self.phase_crouch_end)
            return 0.1 + 0.8 * progress
        elif phase < self.phase_tuck_end:
            progress = (phase - self.phase_launch_end) / (self.phase_tuck_end - self.phase_launch_end)
            return 0.9 + 2.2 * progress
        else:
            progress = (phase - self.phase_tuck_end) / (self.phase_landing_end - self.phase_tuck_end)
            return 3.1 + 3.0 * progress