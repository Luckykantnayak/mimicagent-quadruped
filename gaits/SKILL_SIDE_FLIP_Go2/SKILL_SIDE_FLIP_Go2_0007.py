from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SIDE_FLIP_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Side flip maneuver: 360-degree roll rotation as a pure in-place rotation.
    
    Phase structure:
      [0.0, 0.25]   initial_rotation: roll 0° → ~90°, base begins rotating
      [0.25, 0.75]  inverted_rotation: roll ~90° → ~270°, pass through inverted
      [0.75, 1.0]   completion_rotation: roll ~270° → 360°, return to upright
    
    Base motion: continuous roll rotation with minimal vertical excursion
    Leg motion: reposition continuously in body frame to maintain kinematic validity
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Moderate frequency for controlled in-place rotation
        
        # Base foot positions (body frame, nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.roll_rate_peak = 2.0 * np.pi * 0.85  # Peak roll rate to achieve 360° rotation
        self.vertical_lift_max = 0.35  # Maximum upward velocity (m/s) for modest height adjustment
        self.target_height_offset = 0.12  # Target vertical displacement above nominal (m)
        self.leg_tuck_height = 0.15  # Leg retraction during rotation
        self.leg_lateral_spread = 0.08  # Lateral leg repositioning

    def update_base_motion(self, phase, dt):
        """
        Integrate base pose using phase-dependent velocity commands.
        Vertical motion is minimal, only to accommodate rotation geometry.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.25: initial rotation, roll 0° → 90°
        if phase < 0.25:
            progress = phase / 0.25
            # Smooth ramp-up of roll rate using cubic easing
            roll_ramp = 3.0 * progress**2 - 2.0 * progress**3
            roll_rate = self.roll_rate_peak * roll_ramp
            # Slight upward velocity to elevate base as rotation begins
            vz = self.vertical_lift_max * np.sin(progress * np.pi)
        
        # Phase 0.25-0.75: inverted rotation, roll 90° → 270°
        elif phase < 0.75:
            progress = (phase - 0.25) / 0.5
            # Sustained roll rate through inverted phase
            roll_rate = self.roll_rate_peak
            # Vertical velocity oscillates to maintain modest height during inversion
            # Small downward bias to prevent excessive height accumulation
            vz = -0.15 * np.sin(progress * np.pi) - 0.1
        
        # Phase 0.75-1.0: completion rotation, roll 270° → 360°
        else:
            progress = (phase - 0.75) / 0.25
            # Smooth ramp-down of roll rate using cubic easing
            roll_decay = 1.0 - (3.0 * progress**2 - 2.0 * progress**3)
            roll_rate = self.roll_rate_peak * roll_decay
            # Downward velocity to return base to nominal height
            vz = -self.vertical_lift_max * np.sin(progress * np.pi)
        
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
        Compute foot position in body frame throughout the flip.
        Legs reposition smoothly to maintain kinematic validity as body rotates.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine leg side (left vs right) for symmetric repositioning
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        lateral_sign = 1.0 if is_left else -1.0
        
        # Determine leg front/rear for phase-dependent timing
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Phase 0.0-0.25: initial rotation, begin leg retraction and repositioning
        if phase < 0.25:
            progress = phase / 0.25
            # Smooth cubic retraction
            retract = 3.0 * progress**2 - 2.0 * progress**3
            foot[2] += self.leg_tuck_height * retract
            # Begin lateral repositioning
            foot[1] += lateral_sign * self.leg_lateral_spread * retract
        
        # Phase 0.25-0.75: inverted rotation, legs tucked and repositioned
        elif phase < 0.75:
            progress = (phase - 0.25) / 0.5
            # Maintain full tuck
            foot[2] += self.leg_tuck_height
            # Transition lateral positioning smoothly through inversion
            # Use cosine for smooth transition from one extreme to the other
            lateral_factor = np.cos(progress * np.pi)
            foot[1] += lateral_sign * self.leg_lateral_spread * lateral_factor
            # Add slight forward/backward oscillation to clear rotation geometry
            if is_front:
                foot[0] += 0.03 * np.sin(progress * np.pi)
            else:
                foot[0] -= 0.03 * np.sin(progress * np.pi)
        
        # Phase 0.75-1.0: completion rotation, extend legs back to stance
        else:
            progress = (phase - 0.75) / 0.25
            # Smooth cubic extension back to nominal
            extend = 1.0 - (3.0 * progress**2 - 2.0 * progress**3)
            foot[2] += self.leg_tuck_height * extend
            # Return legs to nominal lateral position
            foot[1] += lateral_sign * self.leg_lateral_spread * (extend - 1.0)
        
        return foot