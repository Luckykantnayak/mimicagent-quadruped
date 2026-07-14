from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_WINDMILL_ROTATION_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Windmill rotation walk gait.
    
    - Right-side legs (FR, RR) execute synchronized windmill circular trajectories
      from phase 0.0-0.5 (swing), then stance from 0.5-1.0
    - Left-side legs (FL, RL) are offset by 0.5 phase, executing stance from 0.0-0.5
      and windmill swing from 0.5-1.0
    - Base moves forward with modulated velocity aligned to stance phases
    - Large vertical circular trajectories create distinctive windmill visual effect
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.6  # Lower frequency for smoother windmill rotation
        
        # Windmill trajectory parameters
        self.windmill_radius_vertical = 0.18  # Vertical radius of windmill arc
        self.windmill_radius_horizontal = 0.12  # Horizontal radius of windmill arc
        self.step_height = self.windmill_radius_vertical  # Max height above neutral
        
        # Base foot positions (neutral stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for side-alternating windmill pattern
        # Right-side legs (FR, RR): phase 0.0
        # Left-side legs (FL, RL): phase 0.5
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('F') and 'R' in leg:  # FR
                self.phase_offsets[leg] = 0.0
            elif leg.startswith('R') and 'R' in leg:  # RR
                self.phase_offsets[leg] = 0.0
            elif leg.startswith('F') and 'L' in leg:  # FL
                self.phase_offsets[leg] = 0.5
            elif leg.startswith('R') and 'L' in leg:  # RL
                self.phase_offsets[leg] = 0.5
        
        # Stance phase duration (50% of cycle for each side)
        self.stance_duration = 0.5
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base velocity parameters
        self.vx_base = 0.4  # Base forward velocity
        self.vx_modulation = 0.15  # Velocity modulation amplitude

    def update_base_motion(self, phase, dt):
        """
        Update base with modulated forward velocity.
        Velocity peaks during stance phases when legs can provide propulsion.
        """
        # Modulate velocity to align with alternating stance phases
        # Higher velocity during left stance (phase 0.0-0.5) and right stance (phase 0.5-1.0)
        # Use smooth sinusoidal modulation with 2x frequency to create two peaks per cycle
        vx = self.vx_base + self.vx_modulation * np.abs(np.sin(2 * np.pi * phase))
        
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position using windmill circular trajectory.
        
        Windmill cycle:
        - Swing phase (0.0-0.5 of leg cycle): Large vertical circular arc
          * 0.0-0.25: Ascend from rear-low to forward-high
          * 0.25-0.5: Descend from forward-high to front contact point
        - Stance phase (0.5-1.0 of leg cycle): Ground contact with rearward motion
          * Foot slides rearward in body frame as base moves forward
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_phase < 0.5:  # Swing phase - windmill circular trajectory
            # Map leg_phase [0.0, 0.5] to circular trajectory parameter [0, 2*pi]
            # Start at bottom-rear, go up-forward, reach apex, descend to front contact
            swing_progress = leg_phase / 0.5  # Normalize to [0, 1]
            
            # Circular parametrization: angle from -pi/2 (bottom) through 0 (front) to pi/2 (top) back to 3*pi/2
            # We want: start rear-low -> forward-high (apex at 0.5) -> front-low
            angle = -np.pi/2 + np.pi * swing_progress  # Goes from -pi/2 to pi/2
            
            # Windmill circular motion
            # X: forward during ascent, then back to contact point
            # Z: vertical circular component
            foot[0] += self.windmill_radius_horizontal * np.sin(angle)
            foot[2] += self.windmill_radius_vertical * (1.0 + np.cos(angle))  # Offset to ensure always above ground
            
        else:  # Stance phase (0.5-1.0) - ground contact with rearward slide
            stance_progress = (leg_phase - 0.5) / 0.5  # Normalize to [0, 1]
            
            # Foot moves rearward in body frame as base advances forward
            # Start from forward contact position, slide to rear
            foot[0] += self.windmill_radius_horizontal * (1.0 - 2.0 * stance_progress)
            # Keep foot on ground (z = 0 offset from base position)
            foot[2] = self.base_feet_pos_body[leg_name][2]
        
        return foot