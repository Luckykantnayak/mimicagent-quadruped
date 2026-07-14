from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_SIDE_FLIP_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Side flip maneuver: 360-degree roll rotation about longitudinal body axis.
    
    - Fully aerial maneuver with no ground contact throughout phase cycle
    - Base executes continuous roll rotation (0° → 90° → 180° → 270° → 360°)
    - All legs retract and reposition in body frame to maintain joint feasibility
    - Minimal linear displacement (in-place flip)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for controlled flip execution
        
        # Base foot positions (nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Roll rate tuned to complete 360° over one phase cycle
        # Angular displacement = roll_rate * duration
        # For freq=0.5, period T=2.0s, need 2π rad total
        # roll_rate = 2π / T = π rad/s
        self.roll_rate = np.pi  # rad/s
        
        # Linear velocity parameters for altitude management
        self.vz_up = 0.5    # Initial upward velocity (phase 0.0-0.25)
        self.vz_down = -0.5  # Downward velocity to return (phase 0.75-1.0)
        
        # Leg retraction parameters
        self.retract_distance = 0.15  # How much to pull legs toward body centerline
        self.retract_height = 0.10    # Additional z-offset during retraction
        
        # Identify left/right legs for symmetric motion
        self.left_legs = [leg for leg in leg_names if leg.startswith('FL') or leg.startswith('RL')]
        self.right_legs = [leg for leg in leg_names if leg.startswith('FR') or leg.startswith('RR')]

    def update_base_motion(self, phase, dt):
        """
        Update base with roll rotation and minimal linear velocity.
        
        Phase 0.0-0.25: Initiate with upward velocity and positive roll rate
        Phase 0.25-0.75: Sustain roll rate, zero linear velocity
        Phase 0.75-1.0: Complete roll with downward velocity
        """
        
        # Linear velocity profile
        if phase < 0.25:
            # Initiation: upward velocity for aerial clearance
            vz = self.vz_up * (1.0 - phase / 0.25)  # Ramp down to zero
        elif phase < 0.75:
            # Inversion transition: no linear velocity (in-place)
            vz = 0.0
        else:
            # Completion: downward velocity to return to original altitude
            vz = self.vz_down * ((phase - 0.75) / 0.25)  # Ramp from zero
        
        # Angular velocity: constant roll rate throughout
        # Slight modulation for smoother completion
        if phase < 0.1:
            # Smooth initiation
            roll_rate = self.roll_rate * (phase / 0.1)
        elif phase > 0.9:
            # Smooth completion
            roll_rate = self.roll_rate * ((1.0 - phase) / 0.1)
        else:
            roll_rate = self.roll_rate
        
        # Set velocity commands (world frame)
        self.vel_world = np.array([0.0, 0.0, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
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
        Compute foot position in body frame throughout flip.
        
        Legs retract during initiation, track body frame through inversion,
        and extend back to nominal stance during completion.
        """
        
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine lateral offset direction (left vs right)
        if leg_name in self.left_legs:
            lateral_sign = 1.0  # Left legs have positive y in body frame
        else:
            lateral_sign = -1.0  # Right legs have negative y
        
        # Phase-dependent leg motion
        if phase < 0.25:
            # Initiation: retract legs toward body centerline
            progress = phase / 0.25
            retract_factor = np.sin(progress * np.pi / 2)  # Smooth 0→1
            
            foot = base_foot.copy()
            # Pull foot inward (reduce lateral distance)
            foot[1] *= (1.0 - retract_factor * 0.6)
            # Pull forward/backward toward center
            foot[0] *= (1.0 - retract_factor * 0.4)
            # Lift foot slightly
            foot[2] += self.retract_height * retract_factor
            
        elif phase < 0.75:
            # Inversion transition: maintain retracted position
            # Add sinusoidal motion to simulate active repositioning through rotation
            mid_phase = (phase - 0.25) / 0.5
            oscillation = np.sin(mid_phase * 2 * np.pi)
            
            foot = base_foot.copy()
            foot[1] *= 0.4  # Retracted lateral position
            foot[0] *= 0.6  # Retracted longitudinal position
            foot[2] += self.retract_height
            
            # Add cyclic motion to track body frame through inverted config
            foot[0] += 0.03 * oscillation  # Small fore-aft oscillation
            foot[2] += 0.02 * np.abs(oscillation)  # Vertical adjustment
            
        else:
            # Completion: extend legs back to nominal stance
            progress = (phase - 0.75) / 0.25
            extend_factor = np.sin(progress * np.pi / 2)  # Smooth 0→1
            
            # Interpolate from retracted to nominal
            retracted_foot = base_foot.copy()
            retracted_foot[1] *= 0.4
            retracted_foot[0] *= 0.6
            retracted_foot[2] += self.retract_height
            
            foot = retracted_foot + extend_factor * (base_foot - retracted_foot)
        
        return foot