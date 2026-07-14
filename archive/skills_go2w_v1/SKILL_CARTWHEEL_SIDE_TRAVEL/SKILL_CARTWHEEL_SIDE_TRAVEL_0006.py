from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_CARTWHEEL_SIDE_TRAVEL_MotionGenerator(BaseMotionGenerator):
    """
    Sideways cartwheel motion with continuous roll rotation and lateral displacement.
    
    The robot performs a cartwheel by:
    - Rolling 360 degrees around its longitudinal (x) axis over one phase cycle
    - Translating laterally (y-direction) throughout the motion
    - Alternating leg support with overlap periods to prevent all-feet-airborne
    - Legs trace conservative arcs in body frame during swing phases
    - Base height adjusts dynamically with roll angle to prevent ground penetration
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.4  # Moderate frequency for controlled cartwheel
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Cartwheel motion parameters
        self.lateral_velocity = 0.4  # Moderate lateral velocity
        self.roll_rate_peak = 1.6 * np.pi  # Slightly reduced for better contact timing
        
        # Conservative swing trajectory parameters
        self.swing_arc_radius = 0.15  # Reduced radius to stay within joint limits
        self.swing_height_max = 0.18  # Reduced height for conservative swing
        
        # Extended overlap phase boundaries to prevent simultaneous airtime
        self.phase_initial_stance_end = 0.22
        self.phase_left_swing_start = 0.25
        self.phase_left_swing_end = 0.45
        self.phase_inversion_overlap_end = 0.55
        self.phase_right_swing_start = 0.58
        self.phase_right_swing_end = 0.78
        self.phase_final_stance_start = 0.82
        
        # Base width for height compensation
        self.base_width = 0.3  # Approximate lateral extent of robot

    def update_base_motion(self, phase, dt):
        """
        Update base motion with lateral velocity, roll rate, and dynamic height compensation.
        Height varies with roll angle to prevent ground penetration during inversion.
        """
        
        # Lateral velocity profile
        if phase < 0.8:
            vy = self.lateral_velocity
        else:
            progress = (phase - 0.8) / 0.2
            vy = self.lateral_velocity * (1.0 - progress)
        
        # Roll rate profile with smooth ramping
        if phase < 0.2:
            progress = phase / 0.2
            roll_rate = self.roll_rate_peak * 0.6 * smooth_transition(progress, 0.0, 1.0)
        elif phase < 0.5:
            roll_rate = self.roll_rate_peak
        elif phase < 0.8:
            roll_rate = self.roll_rate_peak * 0.85
        else:
            progress = (phase - 0.8) / 0.2
            roll_rate = self.roll_rate_peak * 0.85 * (1.0 - smooth_transition(progress, 0.0, 1.0))
        
        # Set velocity commands
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
        
        # Apply dynamic height compensation based on current roll angle
        roll, _, _ = quat_to_euler(self.root_quat)
        roll_normalized = np.abs(np.sin(roll))
        height_offset = self.base_width * 0.5 * roll_normalized
        self.root_pos[2] += height_offset

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with roll-angle-aware stance positioning
        and conservative swing arcs. Extended overlap periods prevent all-feet-airborne.
        """
        
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        # Get current roll angle for stance adjustments
        roll, _, _ = quat_to_euler(self.root_quat)
        roll_deg = np.degrees(roll) % 360
        
        # Determine leg side
        is_left_leg = leg_name in ['FL', 'RL']
        is_right_leg = leg_name in ['FR', 'RR']
        
        # LEFT LEGS (FL, RL) state machine with overlap
        if is_left_leg:
            if phase < self.phase_initial_stance_end:
                # Initial stance: all feet grounded
                return self._apply_stance_adjustment(foot_base, roll_deg, is_left_leg)
            
            elif phase < self.phase_left_swing_start:
                # Pre-swing transition: blend to swing
                blend = (phase - self.phase_initial_stance_end) / (self.phase_left_swing_start - self.phase_initial_stance_end)
                stance_pos = self._apply_stance_adjustment(foot_base, roll_deg, is_left_leg)
                swing_pos = self._compute_swing_arc(foot_base, 0.0, is_left_leg)
                return stance_pos * (1.0 - blend) + swing_pos * blend
            
            elif phase < self.phase_left_swing_end:
                # Active swing phase
                swing_progress = (phase - self.phase_left_swing_start) / (self.phase_left_swing_end - self.phase_left_swing_start)
                return self._compute_swing_arc(foot_base, swing_progress, is_left_leg)
            
            elif phase < self.phase_inversion_overlap_end:
                # Post-swing transition to stance (overlap with right stance)
                blend = (phase - self.phase_left_swing_end) / (self.phase_inversion_overlap_end - self.phase_left_swing_end)
                swing_pos = self._compute_swing_arc(foot_base, 1.0, is_left_leg)
                stance_pos = self._apply_stance_adjustment(foot_base, roll_deg, is_left_leg)
                return swing_pos * (1.0 - blend) + stance_pos * blend
            
            else:
                # Stance for remainder of cycle
                return self._apply_stance_adjustment(foot_base, roll_deg, is_left_leg)
        
        # RIGHT LEGS (FR, RR) state machine with overlap
        elif is_right_leg:
            if phase < self.phase_inversion_overlap_end:
                # Stance through inversion (including overlap with left legs)
                return self._apply_stance_adjustment(foot_base, roll_deg, is_left_leg)
            
            elif phase < self.phase_right_swing_start:
                # Pre-swing transition
                blend = (phase - self.phase_inversion_overlap_end) / (self.phase_right_swing_start - self.phase_inversion_overlap_end)
                stance_pos = self._apply_stance_adjustment(foot_base, roll_deg, is_left_leg)
                swing_pos = self._compute_swing_arc(foot_base, 0.0, is_left_leg)
                return stance_pos * (1.0 - blend) + swing_pos * blend
            
            elif phase < self.phase_right_swing_end:
                # Active swing phase
                swing_progress = (phase - self.phase_right_swing_start) / (self.phase_right_swing_end - self.phase_right_swing_start)
                return self._compute_swing_arc(foot_base, swing_progress, is_left_leg)
            
            elif phase < self.phase_final_stance_start:
                # Post-swing transition to stance
                blend = (phase - self.phase_right_swing_end) / (self.phase_final_stance_start - self.phase_right_swing_end)
                swing_pos = self._compute_swing_arc(foot_base, 1.0, is_left_leg)
                stance_pos = self._apply_stance_adjustment(foot_base, roll_deg, is_left_leg)
                return swing_pos * (1.0 - blend) + stance_pos * blend
            
            else:
                # Final stance: all feet grounded
                return self._apply_stance_adjustment(foot_base, roll_deg, is_left_leg)
        
        return foot_base

    def _apply_stance_adjustment(self, foot_base, roll_deg, is_left_leg):
        """
        Adjust stance foot position based on current roll angle to ensure
        ground contact is kinematically feasible throughout cartwheel rotation.
        """
        foot = foot_base.copy()
        
        # Apply z-offset based on roll angle to keep feet within reachable workspace
        # When inverted (roll near 180°), bring feet closer to base in body frame
        if 45 < roll_deg < 135:
            # Rolling right, approaching 90° - reduce z magnitude
            z_scale = 0.7 + 0.3 * np.cos(np.radians(roll_deg - 90))
            foot[2] *= z_scale
        elif 135 <= roll_deg < 225:
            # Inverted region (90-180°) - significantly reduce z magnitude
            z_scale = 0.5 + 0.2 * np.cos(np.radians(roll_deg - 180))
            foot[2] *= z_scale
        elif 225 <= roll_deg < 315:
            # Rolling left, approaching 270° - reduce z magnitude
            z_scale = 0.7 + 0.3 * np.cos(np.radians(roll_deg - 270))
            foot[2] *= z_scale
        
        return foot

    def _compute_swing_arc(self, foot_base, progress, is_left_leg):
        """
        Compute conservative circular arc trajectory in body frame during swing.
        Reduced extents to stay within joint limits during inverted orientations.
        """
        foot = foot_base.copy()
        
        # Smooth arc progression
        arc_angle = np.pi * smooth_transition(progress, 0.0, 1.0)
        
        # Conservative lateral offset
        y_offset = self.swing_arc_radius * (1.0 - np.cos(arc_angle))
        
        # Conservative vertical offset with smooth profile
        z_offset = self.swing_height_max * np.sin(arc_angle)
        
        # Apply offsets based on leg side
        if foot_base[1] > 0:  # Left side
            foot[1] -= y_offset * 0.8  # Reduced lateral motion
        else:  # Right side
            foot[1] += y_offset * 0.8
        
        # Reduce z-offset magnitude to stay within joint limits
        foot[2] += z_offset * 0.9
        
        return foot