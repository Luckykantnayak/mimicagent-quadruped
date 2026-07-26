from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_WINDMILL_ROTATION_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Windmill rotation walk gait.
    
    Right-side legs (FR, RR) and left-side legs (FL, RL) execute constrained vertical 
    circular motions in alternating phases. One side rotates through the air while 
    the other provides ground support and forward push.
    
    Phase offsets:
    - Right side (FR, RR): phase offset 0.0
    - Left side (FL, RL): phase offset 0.5 (anti-phase for guaranteed stance overlap)
    
    Base motion includes forward velocity, vertical oscillation, and roll oscillation 
    to shift weight between supporting sides.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.6  # Cycle frequency (Hz)
        
        # Windmill rotation parameters - constrained to safe kinematic workspace
        self.windmill_radius = 0.055  # Reduced radius for joint safety
        self.windmill_center_height = -0.26  # Lowered center to keep arc in lower workspace
        self.windmill_center_forward = 0.07  # Reduced forward offset to stay near hip
        
        # Base foot positions (used as reference for lateral positioning)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets: right side synchronized, left side offset by 0.5 for anti-phase
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('F') and 'R' in leg or leg.startswith('R') and 'R' in leg:
                # Right side legs (FR, RR)
                self.phase_offsets[leg] = 0.0
            else:
                # Left side legs (FL, RL)
                self.phase_offsets[leg] = 0.5
        
        # Duty cycle parameters - extended stance for continuous ground contact
        self.swing_duty = 0.35  # Swing occupies 35% of cycle
        self.stance_duty = 0.65  # Stance occupies 65% of cycle (guarantees 15% overlap)
        self.step_length = 0.16  # Forward stride length during stance
        
        # Base motion parameters
        self.vx_forward = 0.28  # Mean forward velocity
        self.vz_amp = 0.04  # Vertical oscillation for stability
        self.roll_amp = 0.10  # Roll oscillation amplitude (radians)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base motion with forward velocity, vertical oscillation, and roll oscillation.
        
        Roll oscillates to shift weight between left and right support sides.
        Vertical motion coordinates with leg transitions to facilitate ground contact.
        """
        
        # Forward velocity - constant
        vx = self.vx_forward
        
        # Vertical velocity - gentle sinusoidal oscillation
        vz = self.vz_amp * np.sin(2 * np.pi * phase * 2.0)
        
        # Roll rate oscillation - smooth continuous function
        roll_rate = self.roll_amp * 2 * np.pi * self.freq * np.cos(2 * np.pi * phase * 2.0)
        
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
        Compute foot position using constrained windmill circular trajectory for swing phase
        and linear rearward motion for stance phase.
        
        Swing phase (0.0 to swing_duty): constrained circular arc motion in lower workspace
        Stance phase (swing_duty to 1.0): rearward linear motion with ground contact
        """
        
        # Apply leg-specific phase offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Get base foot position for lateral (y) reference
        base_foot = self.base_feet_pos_body[leg_name]
        foot = np.zeros(3)
        foot[1] = base_foot[1]  # Lateral position remains constant
        
        if leg_phase < self.swing_duty:
            # Swing phase: constrained windmill circular arc motion
            # Angular range from -50 to +50 degrees (100-degree arc centered on downward vertical)
            # This keeps motion in safe lower workspace
            swing_progress = leg_phase / self.swing_duty
            
            angle_start = -50 * np.pi / 180  # Start from rear-low position
            angle_end = 50 * np.pi / 180     # End at forward-low position
            
            # Smooth progression through arc with ease-in-out
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * swing_progress))
            angle = angle_start + (angle_end - angle_start) * smooth_progress
            
            # Circular trajectory in sagittal plane
            foot[0] = self.windmill_center_forward + self.windmill_radius * np.sin(angle)
            foot[2] = self.windmill_center_height + self.windmill_radius * np.cos(angle)
            
        else:
            # Stance phase: linear rearward motion with guaranteed ground contact
            stance_progress = (leg_phase - self.swing_duty) / self.stance_duty
            
            # Smooth blending into and out of stance phase
            blend_in = min(stance_progress / 0.1, 1.0) if stance_progress < 0.1 else 1.0
            blend_out = min((1.0 - stance_progress) / 0.1, 1.0) if stance_progress > 0.9 else 1.0
            stance_blend = blend_in * blend_out
            
            # Foot moves rearward as body moves forward
            foot[0] = self.step_length * (0.5 - stance_progress)
            
            # Ground level - ensure firm contact throughout stance
            ground_height = self.windmill_center_height + self.windmill_radius * np.cos(50 * np.pi / 180)
            foot[2] = ground_height
        
        return foot