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
        
        # Base state (WORLD frame)
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.peak_pitch_rate = 12.0  # rad/s, peak forward pitch angular velocity
        self.peak_vz_up = 2.5  # m/s, peak upward velocity during takeoff
        self.peak_vz_down = -2.0  # m/s, peak downward velocity during descent
        
        # Leg retraction parameters for aerial phase
        self.front_retract_x = 0.15  # Forward retraction distance for front legs
        self.front_retract_z = 0.12  # Upward retraction distance for front legs
        self.rear_retract_x = -0.15  # Rearward retraction distance for rear legs
        self.rear_retract_z = 0.12  # Upward retraction distance for rear legs
        
        # Landing extension
        self.landing_extension_z = -0.05  # Extra downward extension for landing

    def update_base_motion(self, phase, dt):
        """
        Update base motion with pitch rotation and vertical velocity profile
        to execute forward flip trajectory.
        """
        
        # Pitch rate profile: peaks early, decelerates smoothly to zero by landing
        if phase < 0.1:
            # Rapid ramp-up during takeoff
            pitch_rate = self.peak_pitch_rate * (phase / 0.1)
        elif phase < 0.4:
            # Sustained peak rotation through inverted phase
            pitch_rate = self.peak_pitch_rate
        elif phase < 0.75:
            # Deceleration during rotation completion
            pitch_rate = self.peak_pitch_rate * (1.0 - (phase - 0.4) / 0.35)
        else:
            # Final braking for landing
            pitch_rate = self.peak_pitch_rate * 0.1 * (1.0 - (phase - 0.75) / 0.25)
        
        # Vertical velocity profile: ballistic trajectory
        if phase < 0.05:
            # Initial push-off
            vz = self.peak_vz_up * (phase / 0.05)
        elif phase < 0.3:
            # Upward flight with deceleration
            vz = self.peak_vz_up * (1.0 - (phase - 0.05) / 0.25)
        elif phase < 0.35:
            # Apex transition
            vz = 0.0
        elif phase < 0.7:
            # Downward acceleration
            vz = self.peak_vz_down * ((phase - 0.35) / 0.35)
        elif phase < 0.9:
            # Sustained descent
            vz = self.peak_vz_down
        else:
            # Landing braking
            vz = self.peak_vz_down * (1.0 - (phase - 0.9) / 0.1)
        
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
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            if is_front:
                # Front legs retract forward and upward
                foot[0] += self.front_retract_x * smooth_progress
                foot[2] += self.front_retract_z * smooth_progress
            elif is_rear:
                # Rear legs retract rearward and upward
                foot[0] += self.rear_retract_x * smooth_progress
                foot[2] += self.rear_retract_z * smooth_progress
        
        # Phase 0.5-0.85: Aerial extension (preparing for landing)
        elif phase < 0.85:
            progress = (phase - 0.5) / 0.35
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            if is_front:
                # Front legs transition from retracted to extended beneath body
                retract_factor = 1.0 - smooth_progress
                foot[0] += self.front_retract_x * retract_factor
                foot[2] += self.front_retract_z * retract_factor
                # Begin extending downward for landing
                foot[2] += self.landing_extension_z * smooth_progress
            elif is_rear:
                # Rear legs transition from retracted to extended beneath body
                retract_factor = 1.0 - smooth_progress
                foot[0] += self.rear_retract_x * retract_factor
                foot[2] += self.rear_retract_z * retract_factor
                # Begin extending downward for landing
                foot[2] += self.landing_extension_z * smooth_progress
        
        # Phase 0.85-1.0: Landing (stance, absorbing impact)
        else:
            # Legs fully extended beneath body for ground contact
            foot[2] += self.landing_extension_z
        
        return foot