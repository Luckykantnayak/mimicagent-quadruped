from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_WINDMILL_ROTATION_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Windmill rotation walk gait.
    
    Right-side legs (FR, RR) and left-side legs (FL, RL) execute large vertical 
    circular motions in alternating phases. One side rotates through the air while 
    the other provides ground support and forward push.
    
    Phase offsets:
    - Right side (FR, RR): phase offset 0.0
    - Left side (FL, RL): phase offset 0.5
    
    Base motion includes forward velocity, vertical oscillation, and roll oscillation 
    to shift weight between supporting sides.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.6  # Cycle frequency (Hz) - slower for dramatic windmill motion
        
        # Windmill rotation parameters
        self.windmill_radius = 0.15  # Radius of vertical circular motion
        self.windmill_center_height = -0.25  # Vertical center of windmill circle in body frame
        self.windmill_center_forward = 0.05  # Slight forward offset of windmill center
        
        # Base foot positions (used as reference for lateral positioning)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets: right side synchronized, left side offset by 0.5
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('F') and 'R' in leg or leg.startswith('R') and 'R' in leg:
                # Right side legs (FR, RR)
                self.phase_offsets[leg] = 0.0
            else:
                # Left side legs (FL, RL)
                self.phase_offsets[leg] = 0.5
        
        # Stance phase parameters
        self.stance_duration = 0.5  # Half cycle is stance
        self.step_length = 0.20  # Forward stride length during stance
        
        # Base motion parameters
        self.vx_forward = 0.35  # Mean forward velocity
        self.vz_amp = 0.08  # Vertical oscillation amplitude
        self.roll_amp = 0.15  # Roll oscillation amplitude (radians)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base motion with forward velocity, vertical oscillation, and roll oscillation.
        
        Roll oscillates to shift weight between left and right support sides.
        Vertical motion coordinates with leg transitions.
        """
        
        # Forward velocity - constant
        vx = self.vx_forward
        
        # Vertical velocity - oscillates with phase to coordinate with leg transitions
        # Goes up when right legs lift (phase 0-0.25), down as they land (phase 0.25-0.5)
        # Goes up when left legs lift (phase 0.5-0.75), down as they land (phase 0.75-1.0)
        vz_phase = (phase * 4) % 2  # Creates two cycles per phase
        if vz_phase < 1.0:
            vz = self.vz_amp * np.cos(vz_phase * np.pi)
        else:
            vz = -self.vz_amp * np.cos((vz_phase - 1.0) * np.pi)
        
        # Roll rate oscillation
        # Negative roll (tilt left) when left legs support (phase 0-0.5)
        # Positive roll (tilt right) when right legs support (phase 0.5-1.0)
        # Smooth sinusoidal transition
        roll_rate = self.roll_amp * 2 * np.pi * self.freq * np.cos(2 * np.pi * phase)
        
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position using windmill circular trajectory for swing phase
        and linear rearward motion for stance phase.
        
        Swing phase (0.0-0.5 in leg phase): vertical circular motion
        Stance phase (0.5-1.0 in leg phase): rearward linear motion with ground contact
        """
        
        # Apply leg-specific phase offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Get base foot position for lateral (y) reference
        base_foot = self.base_feet_pos_body[leg_name]
        foot = np.zeros(3)
        foot[1] = base_foot[1]  # Lateral position remains constant
        
        if leg_phase < 0.5:
            # Swing phase: windmill circular motion
            # Angle goes from -90 degrees (bottom rear) to +90 degrees (bottom front)
            # through +180 degrees (top)
            angle = -np.pi/2 + 2 * np.pi * leg_phase
            
            # Circular trajectory in sagittal plane
            foot[0] = self.windmill_center_forward + self.windmill_radius * np.sin(angle)
            foot[2] = self.windmill_center_height + self.windmill_radius * np.cos(angle)
            
        else:
            # Stance phase: linear rearward motion
            stance_progress = (leg_phase - 0.5) / 0.5
            
            # Foot moves rearward as body moves forward
            foot[0] = self.step_length * (0.5 - stance_progress)
            foot[2] = self.windmill_center_height - self.windmill_radius  # Ground level
        
        return foot