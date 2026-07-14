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
        self.freq = 0.5  # Slower frequency for full flip execution
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters - reduced for constraint compliance
        self.crouch_depth = 0.05  # Reduced body lowering during crouch
        self.tuck_retraction = 0.07  # Reduced leg retraction during tuck
        self.extension_length = 0.08  # Reduced leg extension during launch
        
        # Base velocity parameters - reduced to stay within height envelope
        self.launch_vz = 0.8  # Reduced upward velocity during launch
        self.launch_vx = 0.6  # Reduced forward velocity during launch
        self.pitch_rate_max = 2.0 * np.pi  # Reduced pitch rate for 360-degree flip
        
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
            # Gentle downward velocity during crouch
            vz = -0.2 * np.sin(np.pi * progress)
            # Small forward pitch preparation
            pitch_rate = 0.4 * np.sin(np.pi * progress)
            vx = 0.08 * progress
        
        # Phase 1: Extension and launch [0.25, 0.5]
        elif phase < self.phase_launch_end:
            progress = (phase - self.phase_crouch_end) / (self.phase_launch_end - self.phase_crouch_end)
            # Upward impulse concentrated early in phase
            vz = self.launch_vz * np.sin(np.pi * progress * 0.7)
            vx = self.launch_vx * (0.8 + 0.4 * progress)
            # Strong pitch rate for forward rotation
            pitch_rate = self.pitch_rate_max * np.sin(np.pi * progress * 0.6)
        
        # Phase 2: Airborne tuck and rotation [0.5, 0.75]
        elif phase < self.phase_tuck_end:
            progress = (phase - self.phase_launch_end) / (self.phase_tuck_end - self.phase_launch_end)
            # Reduced forward velocity, stronger downward component
            vx = self.launch_vx * (0.7 - 0.4 * progress)
            vz = -1.2 * progress  # Stronger gravity effect to control peak height
            # Sustained pitch rate during tuck
            pitch_rate = self.pitch_rate_max * (0.9 + 0.1 * np.sin(np.pi * progress))
        
        # Phase 3: Landing extension and recovery [0.75, 1.0]
        else:
            progress = (phase - self.phase_tuck_end) / (self.phase_landing_end - self.phase_tuck_end)
            # Decelerate forward velocity
            vx = self.launch_vx * 0.3 * (1.0 - progress)
            # Stronger downward velocity for landing approach
            vz = -1.0 * (1.0 - progress**1.5)
            # Decelerate pitch rate to zero
            pitch_rate = self.pitch_rate_max * 0.3 * (1.0 - progress)**2
        
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
        Compute foot position in body frame based on phase.
        All legs move symmetrically with ground clearance coordination.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine if front or rear leg
        is_front = leg_name.startswith('F')
        
        # Add safety offset to keep feet above ground in initial configuration
        foot[2] += 0.02
        
        # Phase 0: Crouch [0.0, 0.25]
        if phase < self.phase_crouch_end:
            progress = phase / self.phase_crouch_end
            # Keep feet relatively stable while body lowers
            # Minimal upward adjustment in body frame
            foot[2] += 0.01 * progress
            # Reduced inward retraction
            if is_front:
                foot[0] -= 0.015 * progress
            else:
                foot[0] += 0.015 * progress
        
        # Phase 1: Extension and launch [0.25, 0.5]
        elif phase < self.phase_launch_end:
            progress = (phase - self.phase_crouch_end) / (self.phase_launch_end - self.phase_crouch_end)
            # Blend from crouch to extension
            crouch_component = 0.01 * (1.0 - progress)
            # Delayed extension - starts after progress > 0.3
            if progress > 0.3:
                extension_progress = (progress - 0.3) / 0.7
                extension_component = -self.extension_length * extension_progress
            else:
                extension_component = 0.0
            foot[2] += crouch_component + extension_component
            # Reduced outward extension
            if is_front:
                foot[0] -= 0.015 * (1.0 - progress)
                foot[0] += 0.02 * progress
            else:
                foot[0] += 0.015 * (1.0 - progress)
                foot[0] -= 0.02 * progress
        
        # Phase 2: Airborne tuck [0.5, 0.75]
        elif phase < self.phase_tuck_end:
            progress = (phase - self.phase_launch_end) / (self.phase_tuck_end - self.phase_launch_end)
            # Smooth transition from extended to tucked
            extension_component = -self.extension_length * (1.0 - progress)
            tuck_component = self.tuck_retraction * progress
            foot[2] += extension_component + tuck_component
            # Pull feet toward body centerline in x
            x_offset = 0.02 if is_front else -0.02
            foot[0] += x_offset * (1.0 - progress)
            # Minimal lateral compression
            foot[1] *= (1.0 - 0.1 * progress)
        
        # Phase 3: Landing extension [0.75, 1.0]
        else:
            progress = (phase - self.phase_tuck_end) / (self.phase_landing_end - self.phase_tuck_end)
            # Extend from tucked position for landing
            tuck_component = self.tuck_retraction * (1.0 - progress)
            # Stronger landing extension to reach ground
            landing_extension = 0.12 * progress
            foot[2] += tuck_component - landing_extension
            # Return to nominal lateral positions
            foot[1] *= (0.9 + 0.1 * progress)
            # Return to nominal x positions
            x_offset = 0.02 if is_front else -0.02
            foot[0] += x_offset * progress
        
        return foot