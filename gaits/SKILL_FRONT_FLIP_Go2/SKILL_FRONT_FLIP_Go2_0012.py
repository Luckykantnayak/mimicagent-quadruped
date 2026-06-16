from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_FRONT_FLIP_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Front flip motion generator for quadruped robot.
    
    The robot executes a complete 360-degree forward rotation about the pitch axis
    while airborne, with coordinated leg repositioning to maintain kinematic feasibility
    and stable landing on all four feet.
    
    Phase breakdown:
    - [0.0, 0.25]: Takeoff and initial rotation (~90 degrees)
    - [0.25, 0.5]: Inverted transition (~180 degrees total)
    - [0.5, 0.75]: Rotation completion (~270 degrees total)
    - [0.75, 1.0]: Landing and stabilization (~360 degrees total)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time
        self.t = 0.0
        
        # Base state (WORLD frame) - initialize at safe standing height
        self.root_pos = np.array([0.0, 0.0, 0.28])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters - reduced vertical velocities to stay within envelope
        self.peak_pitch_rate = 12.0  # rad/s, peak forward pitch angular velocity
        self.peak_vz_up = 1.6  # m/s, reduced from 2.5 to limit max height
        self.peak_vz_down = -1.5  # m/s, reduced from -2.0 for gentler descent
        
        # Leg retraction parameters for aerial phase
        self.front_retract_x = 0.15  # Forward retraction distance for front legs
        self.front_retract_z = 0.12  # Upward retraction distance for front legs
        self.rear_retract_x = -0.15  # Rearward retraction distance for rear legs
        self.rear_retract_z = 0.12  # Upward retraction distance for rear legs
        
        # Landing extension - changed to small positive to avoid driving feet down
        self.landing_extension_z = 0.02  # Small upward bias for safety margin

    def update_base_motion(self, phase, dt):
        """
        Update base motion with pitch rotation and vertical velocity profile
        to execute forward flip trajectory.
        """
        
        # Pitch rate profile: peaks early, decelerates smoothly to zero by landing
        if phase < 0.1:
            # Rapid ramp-up during takeoff
            pitch_rate = self.peak_pitch_rate * smooth_phase_signal(phase / 0.1)
        elif phase < 0.4:
            # Sustained peak rotation through inverted phase
            pitch_rate = self.peak_pitch_rate
        elif phase < 0.75:
            # Deceleration during rotation completion
            pitch_rate = self.peak_pitch_rate * (1.0 - smooth_phase_signal((phase - 0.4) / 0.35))
        else:
            # Final braking for landing
            pitch_rate = self.peak_pitch_rate * 0.05 * (1.0 - smooth_phase_signal((phase - 0.75) / 0.25))
        
        # Vertical velocity profile: ballistic trajectory with early braking
        if phase < 0.05:
            # Initial push-off with smooth ramp
            vz = self.peak_vz_up * smooth_phase_signal(phase / 0.05)
        elif phase < 0.25:
            # Upward flight with deceleration toward apex
            vz = self.peak_vz_up * (1.0 - smooth_phase_signal((phase - 0.05) / 0.20))
        elif phase < 0.30:
            # Apex transition - brief zero velocity window
            vz = 0.0
        elif phase < 0.65:
            # Downward acceleration to peak descent rate
            vz = self.peak_vz_down * smooth_phase_signal((phase - 0.30) / 0.35)
        elif phase < 0.80:
            # Begin braking earlier to zero velocity before landing contact
            progress = (phase - 0.65) / 0.15
            vz = self.peak_vz_down * (1.0 - smooth_phase_signal(progress))
        else:
            # Velocity fully arrested at landing phase
            vz = 0.0
        
        # Set velocities (no lateral or yaw motion)
        self.vel_world = np.array([0.0, 0.0, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
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
        Compute foot position in BODY frame for given leg and phase.
        
        Front legs (FL, FR): retract forward and upward during aerial phase
        Rear legs (RL, RR): retract rearward and upward during aerial phase
        All legs: extended beneath body during takeoff and landing
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_rear = leg_name.startswith('RL') or leg_name.startswith('RR')
        
        # Phase 0.0-0.1: Takeoff (stance, minimal motion)
        if phase < 0.1:
            # Legs remain in nominal stance position
            pass
        
        # Phase 0.1-0.5: Aerial retraction (inverted transition)
        elif phase < 0.5:
            progress = (phase - 0.1) / 0.4
            smooth_progress = smooth_phase_signal(progress)
            
            if is_front:
                # Front legs retract forward and upward
                foot[0] += self.front_retract_x * smooth_progress
                foot[2] += self.front_retract_z * smooth_progress
            elif is_rear:
                # Rear legs retract rearward and upward
                foot[0] += self.rear_retract_x * smooth_progress
                foot[2] += self.rear_retract_z * smooth_progress
        
        # Phase 0.5-0.75: Aerial extension preparing for landing - maintain some retraction
        elif phase < 0.75:
            progress = (phase - 0.5) / 0.25
            smooth_progress = smooth_phase_signal(progress)
            
            if is_front:
                # Front legs transition from fully retracted to partially retracted
                # Keep legs elevated during descent phase
                retract_factor = 1.0 - 0.6 * smooth_progress  # Only reduce retraction by 60%
                foot[0] += self.front_retract_x * retract_factor
                foot[2] += self.front_retract_z * retract_factor
            elif is_rear:
                # Rear legs transition similarly
                retract_factor = 1.0 - 0.6 * smooth_progress
                foot[0] += self.rear_retract_x * retract_factor
                foot[2] += self.rear_retract_z * retract_factor
        
        # Phase 0.75-0.85: Final approach - complete extension synchronizing with vz=0
        elif phase < 0.85:
            progress = (phase - 0.75) / 0.10
            smooth_progress = smooth_phase_signal(progress)
            
            if is_front:
                # Complete transition to nominal stance with small safety margin
                retract_factor = 0.4 * (1.0 - smooth_progress)
                foot[0] += self.front_retract_x * retract_factor
                foot[2] += self.front_retract_z * retract_factor
                # Add small positive extension for ground clearance safety
                foot[2] += self.landing_extension_z * smooth_progress
            elif is_rear:
                retract_factor = 0.4 * (1.0 - smooth_progress)
                foot[0] += self.rear_retract_x * retract_factor
                foot[2] += self.rear_retract_z * retract_factor
                foot[2] += self.landing_extension_z * smooth_progress
        
        # Phase 0.85-1.0: Landing (stance, ground contact)
        else:
            # Legs at nominal stance with small positive offset for safety
            foot[2] += self.landing_extension_z
        
        return foot