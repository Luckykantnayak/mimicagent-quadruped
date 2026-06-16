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
        self.compression_depth = 0.08  # How much legs compress (body frame z change)
        self.tuck_height = 0.10  # How much legs tuck during flight
        self.forward_step = 0.12  # Forward displacement per hop cycle
        
        # Vertical velocity parameters
        self.vz_launch_max = 1.2  # Peak upward velocity during launch
        self.vz_descent_max = -1.2  # Peak downward velocity during descent
        
        # Forward velocity (constant)
        self.vx_forward = 0.5  # Constant forward speed
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        Vertical velocity oscillates through hop cycle:
        - Compression (0.0-0.2): downward, decreasing to zero
        - Launch (0.2-0.4): upward, rapidly increasing
        - Flight peak (0.4-0.6): upward to downward through zero
        - Descent (0.6-0.8): downward, increasing magnitude
        - Landing (0.8-1.0): downward, decreasing to zero
        
        Forward velocity remains constant throughout.
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

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        
        All legs move identically (synchronized):
        - Compression (0.0-0.2): foot moves up in body frame as leg shortens
        - Launch (0.2-0.4): foot extends down and back, then breaks contact
        - Flight tuck (0.4-0.6): foot retracts up and forward (tucking)
        - Descent extension (0.6-0.8): foot extends down and forward for landing
        - Landing absorption (0.8-1.0): foot moves up in body frame as leg compresses
        """
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if front or rear leg for forward/backward adjustments
        is_front = leg_name.startswith('F')
        
        if phase < 0.2:
            # Compression: foot rises in body frame, moves slightly rearward
            progress = phase / 0.2
            foot[2] += self.compression_depth * progress  # Move up (less negative z)
            foot[0] -= 0.5 * self.forward_step * progress  # Move rearward
            
        elif phase < 0.4:
            # Launch: foot extends down and rearward as leg pushes off
            progress = (phase - 0.2) / 0.2
            # Return from compressed state to extended
            foot[2] += self.compression_depth * (1.0 - progress)
            foot[0] -= 0.5 * self.forward_step * (1.0 - progress * 0.5)
            
        elif phase < 0.6:
            # Flight tuck: foot retracts upward and forward
            progress = (phase - 0.4) / 0.2
            # Smooth tucking motion
            tuck_curve = np.sin(np.pi * progress)
            foot[2] += self.tuck_height * tuck_curve  # Tuck up
            foot[0] += 0.5 * self.forward_step * progress  # Move forward
            
        elif phase < 0.8:
            # Descent extension: foot extends downward and forward for landing
            progress = (phase - 0.6) / 0.2
            # Untuck from peak
            tuck_curve = np.sin(np.pi * (1.0 - progress))
            foot[2] += self.tuck_height * tuck_curve
            # Reach forward to landing position
            foot[0] += 0.5 * self.forward_step * (1.0 + progress)
            
        else:
            # Landing absorption: foot rises in body frame as leg compresses
            progress = (phase - 0.8) / 0.2
            foot[2] += self.compression_depth * progress  # Compress
            foot[0] += 0.5 * self.forward_step  # At forward landing position
        
        return foot