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
        
        # Motion parameters
        self.crouch_depth = 0.08  # How much to lower base during crouch
        self.tuck_retraction = 0.12  # How much to retract legs during tuck
        self.extension_length = 0.10  # Leg extension distance during launch
        
        # Base velocity parameters
        self.launch_vz = 1.5  # Upward velocity during launch
        self.launch_vx = 0.8  # Forward velocity during launch
        self.pitch_rate_max = 3.0 * np.pi  # Peak pitch rate (rad/s) during rotation
        
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
            # Small downward velocity during crouch
            vz = -0.3 * (1.0 - progress)
            # Small forward pitch preparation
            pitch_rate = 0.5 * np.sin(np.pi * progress)
            vx = 0.1 * progress
        
        # Phase 1: Extension and launch [0.25, 0.5]
        elif phase < self.phase_launch_end:
            progress = (phase - self.phase_crouch_end) / (self.phase_launch_end - self.phase_crouch_end)
            # Strong upward and forward velocity
            vz = self.launch_vz * np.sin(np.pi * progress)
            vx = self.launch_vx * (1.0 + progress)
            # Strong positive pitch rate for forward rotation
            pitch_rate = self.pitch_rate_max * np.sin(np.pi * progress * 0.5)
        
        # Phase 2: Airborne tuck and rotation [0.5, 0.75]
        elif phase < self.phase_tuck_end:
            progress = (phase - self.phase_launch_end) / (self.phase_tuck_end - self.phase_launch_end)
            # Decaying forward velocity, gravity pulls down
            vx = self.launch_vx * (1.0 - progress * 0.5)
            vz = -1.0 * progress  # Gravity effect
            # Sustained high pitch rate during tuck (tight moment of inertia)
            pitch_rate = self.pitch_rate_max * (1.0 + 0.3 * progress)
        
        # Phase 3: Landing extension and recovery [0.75, 1.0]
        else:
            progress = (phase - self.phase_tuck_end) / (self.phase_landing_end - self.phase_tuck_end)
            # Decelerate all velocities to zero for stable landing
            vx = self.launch_vx * 0.5 * (1.0 - progress)
            vz = -0.8 * (1.0 - progress**2)  # Downward then decelerate
            # Decelerate pitch rate to zero as approaching upright
            pitch_rate = self.pitch_rate_max * 0.5 * (1.0 - progress)**2
        
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
        All legs move symmetrically.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine if front or rear leg for slight adjustments
        is_front = leg_name.startswith('F')
        
        # Phase 0: Crouch [0.0, 0.25]
        if phase < self.phase_crouch_end:
            progress = phase / self.phase_crouch_end
            # Retract legs inward and upward (crouch)
            foot[2] += self.crouch_depth * progress  # Raise foot relative to body (body lowers)
            # Slight inward retraction in x
            if is_front:
                foot[0] -= 0.03 * progress
            else:
                foot[0] += 0.03 * progress
        
        # Phase 1: Extension and launch [0.25, 0.5]
        elif phase < self.phase_launch_end:
            progress = (phase - self.phase_crouch_end) / (self.phase_launch_end - self.phase_crouch_end)
            # Extend legs downward and outward
            # Start from crouched position
            foot[2] += self.crouch_depth * (1.0 - progress)
            # Add extension component
            foot[2] -= self.extension_length * progress
            # Outward extension in x
            if is_front:
                foot[0] -= 0.03 * (1.0 - progress)
                foot[0] += 0.04 * progress
            else:
                foot[0] += 0.03 * (1.0 - progress)
                foot[0] -= 0.04 * progress
        
        # Phase 2: Airborne tuck [0.5, 0.75]
        elif phase < self.phase_tuck_end:
            progress = (phase - self.phase_launch_end) / (self.phase_tuck_end - self.phase_launch_end)
            # Tuck legs tightly toward body center
            # Transition from extended to tucked
            extension_component = -self.extension_length * (1.0 - progress)
            tuck_component = self.tuck_retraction * progress
            foot[2] += extension_component + tuck_component
            # Pull feet toward body centerline in x
            x_offset = 0.04 if is_front else -0.04
            foot[0] += x_offset * (1.0 - progress)
            # Reduce lateral spread slightly
            foot[1] *= (1.0 - 0.2 * progress)
        
        # Phase 3: Landing extension [0.75, 1.0]
        else:
            progress = (phase - self.phase_tuck_end) / (self.phase_landing_end - self.phase_tuck_end)
            # Extend legs downward to reach for ground
            # Start from tucked position
            tuck_component = self.tuck_retraction * (1.0 - progress)
            foot[2] += tuck_component
            # Extend downward for landing
            landing_extension = 0.05 * progress
            foot[2] -= landing_extension
            # Return to nominal x positions
            foot[1] *= (0.8 + 0.2 * progress)
        
        return foot