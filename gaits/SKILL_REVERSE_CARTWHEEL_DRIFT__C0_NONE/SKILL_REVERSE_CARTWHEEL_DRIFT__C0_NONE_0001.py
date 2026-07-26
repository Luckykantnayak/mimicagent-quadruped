from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERSE_CARTWHEEL_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Reverse cartwheel drift: continuous backward motion with 360-degree roll per cycle.
    
    - Base executes full roll rotation (360 degrees) over one phase cycle
    - Sustained backward velocity throughout
    - Legs alternate overhead extension synchronized with roll angle
    - Left legs (FL, RL) support during phase [0, 0.25], swing during [0.25, 1.0]
    - Right legs (FR, RR) swing during [0, 0.5], support during [0.5, 1.0]
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for dynamic cartwheel motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Cartwheel motion parameters
        self.backward_velocity = -0.8  # Sustained backward drift (negative x)
        self.peak_backward_velocity = -1.2  # Peak during inverted phase
        
        # Roll rate to achieve 360 degrees over one cycle
        # Total rotation = integral of roll_rate over cycle period
        # Period = 1/freq, target rotation = 2*pi radians
        self.base_roll_rate = 2 * np.pi * self.freq  # rad/s for 360 deg per cycle
        self.peak_roll_rate = 1.5 * self.base_roll_rate  # Higher during inverted phase
        
        # Leg extension parameters
        self.lateral_extension = 0.25  # How far legs extend laterally during overhead phase
        self.vertical_extension = 0.35  # How high legs extend during overhead arc
        self.stance_width = 0.15  # Lateral offset during stance
        self.stance_depth = -0.05  # Vertical offset during stance (below base)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base with backward velocity and continuous roll rotation.
        Roll rate peaks during inverted phase, backward velocity peaks at apex.
        """
        # Backward velocity profile: peak during inverted phase [0.25, 0.5]
        if 0.2 <= phase <= 0.6:
            vx = self.peak_backward_velocity
        else:
            vx = self.backward_velocity
        
        # Minor lateral drift for smoothness during transitions
        vy = 0.0
        if 0.0 <= phase < 0.25:
            vy = 0.05 * np.sin(phase * 4 * np.pi)  # Slight right drift
        elif 0.5 <= phase < 0.75:
            vy = -0.05 * np.sin((phase - 0.5) * 4 * np.pi)  # Slight left drift
        
        vz = 0.0
        
        # Roll rate profile: peak during inverted phase, smooth at boundaries
        if 0.15 <= phase <= 0.6:
            # Peak roll rate during inverted phase
            roll_rate = self.peak_roll_rate
        elif phase < 0.15:
            # Ramp up smoothly from cycle start
            ramp = phase / 0.15
            roll_rate = self.base_roll_rate + (self.peak_roll_rate - self.base_roll_rate) * ramp
        elif phase > 0.85:
            # Ramp down smoothly toward cycle end
            ramp = (1.0 - phase) / 0.15
            roll_rate = self.base_roll_rate + (self.peak_roll_rate - self.base_roll_rate) * ramp
        else:
            # Transition phases
            roll_rate = self.base_roll_rate
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])  # Roll only
        
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
        Compute foot position in body frame based on cartwheel phase.
        
        Left legs (FL, RL): stance [0, 0.25], overhead swing [0.25, 1.0]
        Right legs (FR, RR): overhead swing [0, 0.5], stance [0.5, 1.0]
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine lateral side
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        if is_left:
            # Left legs: stance [0, 0.25], swing [0.25, 1.0]
            if phase < 0.25:
                # Stance phase: positioned left and down
                foot = self._compute_stance_position(base_pos, -self.stance_width, phase, 0.25)
            else:
                # Swing phase: overhead arc from 0.25 to 1.0
                swing_phase = (phase - 0.25) / 0.75
                foot = self._compute_overhead_arc(base_pos, -1.0, swing_phase)
        
        elif is_right:
            # Right legs: swing [0, 0.5], stance [0.5, 1.0]
            if phase < 0.5:
                # Swing phase: overhead arc from 0.0 to 0.5
                swing_phase = phase / 0.5
                foot = self._compute_overhead_arc(base_pos, 1.0, swing_phase)
            else:
                # Stance phase: positioned right and down
                stance_phase = (phase - 0.5) / 0.5
                foot = self._compute_stance_position(base_pos, self.stance_width, stance_phase, 0.5)
        else:
            foot = base_pos
        
        return foot

    def _compute_stance_position(self, base_pos, lateral_sign, local_phase, duration):
        """
        Compute stance foot position: laterally offset, grounded, with backward slide.
        
        lateral_sign: -1 for left, +1 for right
        local_phase: phase within stance period [0, 1]
        """
        foot = base_pos.copy()
        
        # Lateral offset for support
        foot[1] = lateral_sign * self.stance_width
        
        # Vertical position: grounded
        foot[2] = self.stance_depth
        
        # Backward slide during stance to contribute to backward propulsion
        backward_slide = 0.08
        foot[0] = base_pos[0] - backward_slide * local_phase
        
        return foot

    def _compute_overhead_arc(self, base_pos, lateral_sign, swing_phase):
        """
        Compute overhead arc trajectory for cartwheel swing.
        
        lateral_sign: -1 for left, +1 for right
        swing_phase: normalized phase [0, 1] within swing period
        
        Arc traces from ground -> overhead -> back to ground
        """
        foot = base_pos.copy()
        
        # Circular arc parameterization
        # At swing_phase = 0: starting position (near ground)
        # At swing_phase = 0.5: maximum overhead extension
        # At swing_phase = 1.0: returning to ground
        
        angle = np.pi * swing_phase  # 0 to pi over swing
        
        # Lateral extension: maximum at apex (swing_phase = 0.5)
        foot[1] = lateral_sign * self.lateral_extension * np.sin(angle)
        
        # Vertical extension: arc reaches peak at apex
        # Use sin for smooth arc: 0 -> peak -> 0
        foot[2] = self.vertical_extension * np.sin(angle)
        
        # Forward/backward position: slight forward reach during overhead
        # to maintain body-frame consistency during roll
        foot[0] = base_pos[0] + 0.05 * np.sin(2 * angle)
        
        return foot