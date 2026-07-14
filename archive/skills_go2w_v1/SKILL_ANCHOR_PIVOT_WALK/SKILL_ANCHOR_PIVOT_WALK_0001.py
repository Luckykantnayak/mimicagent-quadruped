from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ANCHOR_PIVOT_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Anchor-pivot walking gait with sequential single-leg support.
    
    Motion pattern:
    - Each leg takes turns as a stationary anchor (pivot point)
    - While one leg is anchored, the other three swing forward in arcs
    - Base rotates around the anchor leg while translating forward
    - Sequence: FL anchor → FR anchor → RL anchor → RR anchor
    
    Phase structure (4 sub-phases of 0.25 duration each):
    - [0.00, 0.25]: FL anchored, FR/RL/RR swing
    - [0.25, 0.50]: FR anchored, FL/RL/RR swing
    - [0.50, 0.75]: RL anchored, FL/FR/RR swing
    - [0.75, 1.00]: RR anchored, FL/FR/RL swing
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for stability during single-leg support
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.step_length = 0.12  # Forward stride length during swing
        self.step_height = 0.10  # Arc height during swing phase
        self.lateral_offset = 0.03  # Lateral shift toward anchor leg
        
        # Base velocity parameters
        self.vx_base = 0.15  # Forward velocity magnitude
        self.yaw_rate_magnitude = 0.8  # Yaw rotation rate around anchor
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Identify legs by name prefix
        self.FL = None
        self.FR = None
        self.RL = None
        self.RR = None
        for leg in leg_names:
            if leg.startswith('FL'):
                self.FL = leg
            elif leg.startswith('FR'):
                self.FR = leg
            elif leg.startswith('RL'):
                self.RL = leg
            elif leg.startswith('RR'):
                self.RR = leg

    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent velocity commands.
        
        Each quarter-phase corresponds to one anchor leg:
        - Yaw rate alternates sign to create balanced pivoting
        - Lateral velocity shifts CoM toward current anchor
        - Forward velocity maintains net forward progress
        """
        
        # Determine current sub-phase and anchor leg
        if phase < 0.25:
            # FL anchor phase
            vx = self.vx_base
            vy = -self.lateral_offset * 2.0  # Shift left toward FL
            yaw_rate = self.yaw_rate_magnitude  # Counter-clockwise
        elif phase < 0.5:
            # FR anchor phase
            vx = self.vx_base
            vy = self.lateral_offset * 2.0  # Shift right toward FR
            yaw_rate = -self.yaw_rate_magnitude  # Clockwise
        elif phase < 0.75:
            # RL anchor phase
            vx = self.vx_base
            vy = -self.lateral_offset * 2.0  # Shift left toward RL
            yaw_rate = self.yaw_rate_magnitude  # Counter-clockwise
        else:
            # RR anchor phase
            vx = self.vx_base
            vy = self.lateral_offset * 2.0  # Shift right toward RR
            yaw_rate = -self.yaw_rate_magnitude  # Clockwise
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate base pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on current phase.
        
        Each leg alternates between:
        - Stance (anchor): foot remains at fixed body-frame position
        - Swing: foot executes arc trajectory with forward advancement
        
        Swing timing varies per leg according to the anchor sequence.
        """
        
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        # FL: stance [0.0, 0.25], swing [0.25, 1.0]
        if leg_name == self.FL:
            if phase < 0.25:
                # Stance phase - anchor position
                return foot_base
            else:
                # Swing phase from 0.25 to 1.0
                swing_progress = (phase - 0.25) / 0.75
                return self._compute_swing_trajectory(foot_base, swing_progress)
        
        # FR: swing [0.0, 0.25], stance [0.25, 0.5], swing [0.5, 1.0]
        elif leg_name == self.FR:
            if phase < 0.25:
                # Tail end of swing from previous cycle
                swing_progress = (phase + 0.5) / 0.75  # Continue from where it left off
                return self._compute_swing_trajectory(foot_base, swing_progress)
            elif phase < 0.5:
                # Stance phase - anchor position
                return foot_base
            else:
                # Swing phase from 0.5 to 1.0 (continues into next cycle)
                swing_progress = (phase - 0.5) / 0.75
                return self._compute_swing_trajectory(foot_base, swing_progress)
        
        # RL: swing [0.0, 0.5], stance [0.5, 0.75], swing [0.75, 1.0]
        elif leg_name == self.RL:
            if phase < 0.5:
                # Extended swing covering two anchor phases
                swing_progress = (phase + 0.25) / 0.75
                return self._compute_swing_trajectory(foot_base, swing_progress)
            elif phase < 0.75:
                # Stance phase - anchor position
                return foot_base
            else:
                # Swing phase begins (continues into next cycle)
                swing_progress = (phase - 0.75) / 0.75
                return self._compute_swing_trajectory(foot_base, swing_progress)
        
        # RR: swing [0.0, 0.75], stance [0.75, 1.0]
        elif leg_name == self.RR:
            if phase < 0.75:
                # Extended swing covering three anchor phases
                swing_progress = phase / 0.75
                return self._compute_swing_trajectory(foot_base, swing_progress)
            else:
                # Stance phase - anchor position
                return foot_base
        
        # Fallback (should not reach here)
        return foot_base

    def _compute_swing_trajectory(self, foot_base, progress):
        """
        Compute swing arc trajectory for a foot.
        
        Args:
            foot_base: Base foot position in body frame
            progress: Swing phase progress in [0, 1]
        
        Returns:
            foot position during swing with forward advancement and vertical arc
        """
        foot = foot_base.copy()
        
        # Forward advancement: move from back to front during swing
        # At progress=0: foot at rear position
        # At progress=1: foot at forward position
        foot[0] += self.step_length * (progress - 0.5)
        
        # Vertical arc: sinusoidal lift and lower
        # Peak height at progress=0.5
        arc_angle = np.pi * progress
        foot[2] += self.step_height * np.sin(arc_angle)
        
        return foot