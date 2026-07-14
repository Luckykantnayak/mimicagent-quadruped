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
    - Left side (FL, RL): phase offset 0.4 (adjusted for stance overlap)
    
    Base motion includes forward velocity, vertical oscillation, and roll oscillation 
    to shift weight between supporting sides.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.6  # Cycle frequency (Hz) - slower for dramatic windmill motion
        
        # Windmill rotation parameters - reduced for joint workspace safety
        self.windmill_radius = 0.09  # Reduced radius to keep trajectory within kinematic limits
        self.windmill_center_height = -0.22  # Raised center to keep arc reachable
        self.windmill_center_forward = 0.11  # Increased forward offset for rear arc clearance
        
        # Base foot positions (used as reference for lateral positioning)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets: right side synchronized, left side offset by 0.4 for stance overlap
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('F') and 'R' in leg or leg.startswith('R') and 'R' in leg:
                # Right side legs (FR, RR)
                self.phase_offsets[leg] = 0.0
            else:
                # Left side legs (FL, RL)
                self.phase_offsets[leg] = 0.4
        
        # Duty cycle parameters - adjusted for continuous ground contact
        self.swing_duty = 0.38  # Swing occupies 38% of cycle
        self.stance_duty = 0.62  # Stance occupies 62% of cycle
        self.step_length = 0.18  # Forward stride length during stance
        
        # Base motion parameters
        self.vx_forward = 0.32  # Mean forward velocity
        self.vz_amp = 0.05  # Reduced vertical oscillation for stability
        self.roll_amp = 0.12  # Roll oscillation amplitude (radians)
        
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
        
        # Vertical velocity - coordinated with transitions to assist touchdown
        # Lower base during transition windows, lift during mid-swing
        # Right side transition around phase 0.0, left side around phase 0.4
        transition_phase_1 = phase  # Right side transition
        transition_phase_2 = (phase - 0.4) % 1.0  # Left side transition
        
        # Create dip around transitions (phase near 0 and 0.4)
        min_dist_to_transition = min(
            min(transition_phase_1, 1.0 - transition_phase_1),
            min(transition_phase_2, 1.0 - transition_phase_2)
        )
        
        # Lower base when close to transition (within 0.15 phase units)
        if min_dist_to_transition < 0.15:
            transition_factor = min_dist_to_transition / 0.15
            vz = -self.vz_amp * 2.0 * (1.0 - transition_factor)
        else:
            # Gentle sinusoidal oscillation during mid-phases
            vz = self.vz_amp * np.sin(2 * np.pi * phase * 2.5)
        
        # Roll rate oscillation - smooth continuous function
        # Tilt toward supporting side
        roll_rate = self.roll_amp * 2 * np.pi * self.freq * np.sin(2 * np.pi * phase * 2.5)
        
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
        
        Swing phase (0.0 to swing_duty): partial vertical circular arc motion
        Stance phase (swing_duty to 1.0): rearward linear motion with ground contact
        """
        
        # Apply leg-specific phase offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Get base foot position for lateral (y) reference
        base_foot = self.base_feet_pos_body[leg_name]
        foot = np.zeros(3)
        foot[1] = base_foot[1]  # Lateral position remains constant
        
        if leg_phase < self.swing_duty:
            # Swing phase: windmill circular arc motion (partial circle, not full 360)
            # Restrict arc from -70 degrees to +110 degrees (180-degree arc)
            # This avoids extreme top and bottom positions
            swing_progress = leg_phase / self.swing_duty
            angle_start = -70 * np.pi / 180  # Start from rear-low position
            angle_end = 110 * np.pi / 180    # End at forward-low position
            angle = angle_start + (angle_end - angle_start) * swing_progress
            
            # Apply smooth acceleration at start and deceleration at end
            blend = swing_progress * 2.0 - 1.0  # -1 to +1
            smoothing = 0.5 + 0.5 * np.tanh(blend * 2.0)
            angle = angle_start + (angle_end - angle_start) * smoothing
            
            # Circular trajectory in sagittal plane
            foot[0] = self.windmill_center_forward + self.windmill_radius * np.sin(angle)
            foot[2] = self.windmill_center_height + self.windmill_radius * np.cos(angle)
            
        else:
            # Stance phase: linear rearward motion with guaranteed ground contact
            stance_progress = (leg_phase - self.swing_duty) / self.stance_duty
            
            # Initial ground contact guarantee (first 15% of stance)
            if stance_progress < 0.15:
                contact_blend = stance_progress / 0.15
            else:
                contact_blend = 1.0
            
            # Foot moves rearward as body moves forward
            foot[0] = self.step_length * (0.5 - stance_progress * contact_blend)
            
            # Ground level - ensure firm contact
            ground_height = self.windmill_center_height - self.windmill_radius
            foot[2] = ground_height
        
        return foot