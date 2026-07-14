from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_CARTWHEEL_SIDE_TRAVEL_MotionGenerator(BaseMotionGenerator):
    """
    Sideways cartwheel motion with continuous roll rotation and lateral displacement.
    
    The robot performs a cartwheel by:
    - Rolling 360 degrees around its longitudinal (x) axis over one phase cycle
    - Translating laterally (y-direction) throughout the motion
    - Alternating leg support: right legs pivot during inversion (0.2-0.5),
      left legs pivot during uprighting (0.5-0.8)
    - Legs trace circular arcs in body frame during swing phases
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for dramatic cartwheel motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Cartwheel motion parameters
        self.lateral_velocity = 0.5  # Lateral (y) velocity for sideways travel
        self.roll_rate_peak = 2.0 * np.pi  # Peak roll rate (rad/s) for 360-degree rotation
        
        # Swing trajectory parameters (circular arc in body frame)
        self.swing_arc_radius = 0.25  # Radius of circular swing path
        self.swing_height_max = 0.3  # Maximum height above nominal during overhead swing
        
        # Phase boundaries for contact transitions
        self.phase_all_stance_end = 0.2
        self.phase_right_pivot_end = 0.5
        self.phase_left_pivot_end = 0.8

    def update_base_motion(self, phase, dt):
        """
        Update base motion with lateral velocity and roll rate varying by phase.
        
        Phase 0.0-0.25: Begin lateral motion and roll
        Phase 0.25-0.5: Peak roll rate for inversion
        Phase 0.5-0.75: Continue roll to upright
        Phase 0.75-1.0: Decelerate and stabilize
        """
        
        # Lateral velocity profile (y-direction)
        if phase < 0.75:
            vy = self.lateral_velocity
        else:
            # Decelerate in final phase
            progress = (phase - 0.75) / 0.25
            vy = self.lateral_velocity * (1.0 - progress)
        
        # Roll rate profile (positive = rolling right/clockwise around x-axis)
        if phase < 0.25:
            # Ramp up roll rate
            progress = phase / 0.25
            roll_rate = self.roll_rate_peak * 0.7 * progress
        elif phase < 0.5:
            # Peak roll rate during inversion
            roll_rate = self.roll_rate_peak
        elif phase < 0.75:
            # Continue roll at moderate rate
            roll_rate = self.roll_rate_peak * 0.8
        else:
            # Decelerate roll to stabilize
            progress = (phase - 0.75) / 0.25
            roll_rate = self.roll_rate_peak * 0.8 * (1.0 - progress)
        
        # Set velocity commands in world frame
        self.vel_world = np.array([0.0, vy, 0.0])
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
        Compute foot position in body frame based on phase and leg group.
        
        Left legs (FL, RL): swing during 0.2-0.5, stance otherwise
        Right legs (FR, RR): stance during 0.2-0.5, swing during 0.5-0.8
        """
        
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left or right leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Left legs (FL, RL) behavior
        if is_left_leg:
            if phase < self.phase_all_stance_end:
                # All stance: maintain nominal position
                return foot_base
            elif phase < self.phase_right_pivot_end:
                # Swing phase: trace overhead arc
                swing_progress = (phase - self.phase_all_stance_end) / (self.phase_right_pivot_end - self.phase_all_stance_end)
                return self._compute_swing_arc(foot_base, swing_progress)
            elif phase < self.phase_left_pivot_end:
                # Stance phase: maintain ground contact
                return foot_base
            else:
                # Return to stance: maintain nominal position
                return foot_base
        
        # Right legs (FR, RR) behavior
        elif is_right_leg:
            if phase < self.phase_all_stance_end:
                # All stance: maintain nominal position
                return foot_base
            elif phase < self.phase_right_pivot_end:
                # Stance phase (pivot): maintain ground contact
                return foot_base
            elif phase < self.phase_left_pivot_end:
                # Swing phase: trace overhead arc
                swing_progress = (phase - self.phase_right_pivot_end) / (self.phase_left_pivot_end - self.phase_right_pivot_end)
                return self._compute_swing_arc(foot_base, swing_progress)
            else:
                # Return to stance: maintain nominal position
                return foot_base
        
        return foot_base

    def _compute_swing_arc(self, foot_base, progress):
        """
        Compute circular arc trajectory in body frame during swing phase.
        
        The foot traces an arc from ground level to overhead and back,
        approximating the path needed to clear the ground during cartwheel rotation.
        
        Args:
            foot_base: nominal foot position in body frame
            progress: swing phase progress in [0, 1]
        
        Returns:
            foot position in body frame along circular arc
        """
        
        foot = foot_base.copy()
        
        # Circular arc: foot moves in y-z plane relative to nominal position
        # Arc angle from 0 to pi (semicircle overhead)
        arc_angle = np.pi * progress
        
        # Lateral offset (y) follows cosine (starts at nominal, moves inward, returns)
        y_offset = self.swing_arc_radius * (1.0 - np.cos(arc_angle))
        
        # Vertical offset (z) follows sine (lifts overhead and returns)
        z_offset = self.swing_height_max * np.sin(arc_angle)
        
        # Apply offsets
        # Y-offset direction depends on which side of body the leg is on
        if foot_base[1] > 0:  # Left side leg (positive y)
            foot[1] -= y_offset
        else:  # Right side leg (negative y)
            foot[1] += y_offset
        
        foot[2] += z_offset
        
        return foot