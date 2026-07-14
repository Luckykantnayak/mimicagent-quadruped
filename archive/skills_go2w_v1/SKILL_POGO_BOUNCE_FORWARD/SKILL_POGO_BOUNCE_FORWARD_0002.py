from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_POGO_BOUNCE_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Pogo-style bounce with all four legs synchronized.
    
    - All legs move together (zero phase offset)
    - Compression -> Launch -> Flight -> Descent -> Landing absorption
    - Base maintains constant forward velocity with oscillating vertical velocity
    - Feet expressed in BODY frame
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0  # 1 Hz hop frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # All legs synchronized - zero phase offset
        self.phase_offsets = {
            leg_names[0]: 0.0,
            leg_names[1]: 0.0,
            leg_names[2]: 0.0,
            leg_names[3]: 0.0,
        }
        
        # Motion parameters
        self.forward_step = 0.12  # Forward displacement per hop cycle
        self.tuck_height = 0.06  # How much legs tuck during flight (reduced)
        
        # Vertical velocity parameters (reduced by ~70% for safety)
        self.vz_launch_max = 0.35  # Peak upward velocity during launch
        self.vz_descent_max = -0.35  # Peak downward velocity during descent
        
        # Forward velocity (constant)
        self.vx_forward = 0.5  # Constant forward speed
        
        # Nominal base height target
        self.nominal_base_height = 0.30
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.nominal_base_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Track base height for foot coordination
        self.current_base_height = self.nominal_base_height

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        Vertical velocity oscillates through hop cycle with reduced magnitudes
        to maintain safe base height envelope (0.2 to 0.45 m).
        """
        
        vx = self.vx_forward
        vy = 0.0
        vz = 0.0
        
        # Compute vertical velocity based on phase
        if phase < 0.2:
            # Compression: downward velocity decreasing to zero
            progress = phase / 0.2
            vz = self.vz_descent_max * (1.0 - progress)
            
        elif phase < 0.4:
            # Launch: upward velocity rapidly increasing
            progress = (phase - 0.2) / 0.2
            vz = self.vz_launch_max * progress
            
        elif phase < 0.6:
            # Flight peak: upward through zero to downward
            progress = (phase - 0.4) / 0.2
            # Sinusoidal transition from positive to negative
            vz = self.vz_launch_max * np.cos(np.pi * progress)
            
        elif phase < 0.8:
            # Descent: downward velocity increasing in magnitude
            progress = (phase - 0.6) / 0.2
            vz = -self.vz_launch_max * progress
            
        else:
            # Landing absorption: downward velocity decreasing to zero
            progress = (phase - 0.8) / 0.2
            vz = self.vz_descent_max * (1.0 - progress)
        
        # Set velocity commands (world frame)
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Track current base height for foot coordination
        self.current_base_height = self.root_pos[2]

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        
        Ground contact phases (0.0-0.4, 0.8-1.0): feet maintain world z=0
        by adjusting body-frame z to compensate for base height changes.
        Flight phase (0.4-0.8): feet tuck and extend for aerial maneuver.
        """
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        base_foot_z = foot[2]  # Nominal body-frame z (negative, pointing down)
        
        # Compute base height deviation from nominal
        base_height_offset = self.current_base_height - self.nominal_base_height
        
        if phase < 0.2:
            # Compression: base lowers, foot stays on ground (world z=0)
            # As base descends, body-frame foot z becomes less negative (leg shortens)
            progress = phase / 0.2
            # Smooth compression curve
            compression_factor = np.sin(0.5 * np.pi * progress)
            
            # Foot adjusts to maintain ground contact as base compresses
            foot[2] = base_foot_z + base_height_offset
            
            # Foot moves slightly rearward during compression
            foot[0] -= 0.3 * self.forward_step * compression_factor
            
        elif phase < 0.4:
            # Launch: explosive extension, foot pushes off ground then breaks contact
            progress = (phase - 0.2) / 0.2
            
            if progress < 0.7:
                # Still in contact, extending
                foot[2] = base_foot_z + base_height_offset
                # Return toward neutral stance
                remaining_compression = 1.0 - progress / 0.7
                foot[0] -= 0.3 * self.forward_step * remaining_compression
            else:
                # Breaking contact, beginning to lift
                liftoff_progress = (progress - 0.7) / 0.3
                # Smooth liftoff curve
                liftoff_curve = np.sin(0.5 * np.pi * liftoff_progress)
                foot[2] = base_foot_z + base_height_offset - 0.02 * liftoff_curve
                foot[0] += 0.1 * self.forward_step * liftoff_progress
            
        elif phase < 0.6:
            # Flight tuck: foot retracts upward and forward (aerial phase)
            progress = (phase - 0.4) / 0.2
            # Smooth tucking motion
            tuck_curve = np.sin(np.pi * progress)
            foot[2] = base_foot_z + self.tuck_height * tuck_curve
            foot[0] += 0.4 * self.forward_step * progress
            
        elif phase < 0.8:
            # Descent extension: foot extends downward and forward for landing
            progress = (phase - 0.6) / 0.2
            # Untuck from peak
            tuck_curve = np.sin(np.pi * (1.0 - progress))
            foot[2] = base_foot_z + self.tuck_height * tuck_curve
            # Reach forward to landing position
            forward_reach = 0.4 + 0.3 * progress
            foot[0] += self.forward_step * forward_reach
            
        else:
            # Landing absorption: foot contacts ground, base compresses
            progress = (phase - 0.8) / 0.2
            # Smooth landing impact absorption
            absorption_curve = np.sin(0.5 * np.pi * progress)
            
            # Foot maintains ground contact as base lowers
            foot[2] = base_foot_z + base_height_offset
            
            # Foot at forward landing position
            foot[0] += 0.5 * self.forward_step
        
        return foot