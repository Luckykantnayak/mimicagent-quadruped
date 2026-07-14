from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ORBIT_TILT_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Circular orbit locomotion with synchronized rhythmic base roll oscillations.
    
    Motion characteristics:
    - Robot travels in a circular path with constant yaw rate
    - Base rolls left and right in sync with orbital position
    - Roll peaks at phases 0.25 (left), 0.5 (right), 0.75 (left)
    - Diagonal leg coordination (trot-like) maintains continuous contact
    - Outer legs extend more than inner legs to support roll tilts
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # One complete orbit cycle per 2 seconds
        
        # Base foot positions in body frame - ensure adequate ground clearance
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        for leg_name in self.base_feet_pos_body:
            self.base_feet_pos_body[leg_name][2] = max(self.base_feet_pos_body[leg_name][2], 0.02)
        
        # Circular orbit parameters
        self.orbit_radius = 1.0
        self.orbit_angular_velocity = 2.0 * np.pi * self.freq
        self.forward_speed = 0.6
        
        # Roll oscillation parameters
        self.roll_amplitude = 0.35  # radians (~20 degrees)
        self.roll_frequency = 2.0 * self.freq
        
        # Leg motion parameters
        self.swing_duration = 0.125
        self.swing_height = 0.06
        self.stance_step_length = 0.15
        self.extension_modulation = 0.020  # Reduced further for safety
        self.min_foot_height = 0.005  # Minimum safe ground clearance
        
        # Wider transition smoothing window for better acceleration continuity
        self.transition_window = 0.05  # Increased from 0.02
        
        # Phase offsets for diagonal coordination
        self.swing_phases = {
            leg_names[0]: [(0.125, 0.25), (0.625, 0.75)],   # FL
            leg_names[1]: [(0.375, 0.5), (0.875, 1.0)],     # FR
            leg_names[2]: [(0.375, 0.5), (0.875, 1.0)],     # RL
            leg_names[3]: [(0.125, 0.25), (0.625, 0.75)],   # RR
        }
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands to create circular orbit with roll oscillation.
        Uses smoothed trajectories to reduce acceleration discontinuities.
        """
        # Circular orbit velocities
        orbit_angle = 2.0 * np.pi * phase
        
        vx = self.forward_speed * np.cos(orbit_angle)
        vy = self.forward_speed * np.sin(orbit_angle)
        vz = 0.0
        
        # Constant yaw rate for circular turning
        yaw_rate = self.orbit_angular_velocity
        
        # Smoothed roll trajectory
        roll_angle = self._compute_smooth_roll_angle(phase)
        roll_rate = self._compute_smooth_roll_rate(phase)
        
        pitch_rate = 0.0
        
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

    def _compute_smooth_roll_angle(self, phase):
        """
        Compute roll angle with smooth transitions.
        """
        roll_angle_phase = 4.0 * np.pi * phase
        roll_angle = self.roll_amplitude * np.sin(roll_angle_phase)
        return roll_angle

    def _compute_smooth_roll_rate(self, phase):
        """
        Compute roll rate with smoothing.
        """
        roll_angle_phase = 4.0 * np.pi * phase
        roll_rate = self.roll_amplitude * 4.0 * np.pi * self.freq * np.cos(roll_angle_phase)
        return roll_rate

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with smooth transitions and proper ground clearance.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is on left or right side
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Get swing state and progress with velocity matching
        in_swing, swing_progress, blend_factor = self._get_swing_state_with_blending(leg_name, phase)
        
        # Compute current roll angle for extension modulation
        roll_angle = self._compute_smooth_roll_angle(phase)
        
        # Compute stance progress using continuous function
        stance_progress = self._compute_continuous_stance_progress(leg_name, phase)
        
        if in_swing:
            # Pure swing phase with velocity-matched boundaries
            foot = self._compute_swing_foot_with_velocity_matching(
                foot, swing_progress, stance_progress, leg_name, phase
            )
        else:
            # Stance phase with proper directional extension
            foot = self._compute_stance_foot_with_extension(
                foot, stance_progress, is_left_leg, roll_angle
            )
        
        # Apply blending if in transition
        if 0.0 < blend_factor < 1.0:
            foot_swing = self._compute_swing_foot_with_velocity_matching(
                self.base_feet_pos_body[leg_name].copy(), swing_progress, stance_progress, leg_name, phase
            )
            foot_stance = self._compute_stance_foot_with_extension(
                self.base_feet_pos_body[leg_name].copy(), stance_progress, is_left_leg, roll_angle
            )
            # Quintic blending for C2 continuity
            blend_smooth = self._quintic_blend(blend_factor)
            foot = foot_stance * (1.0 - blend_smooth) + foot_swing * blend_smooth
        
        # Enforce minimum ground clearance as safety clamp
        foot[2] = max(foot[2], self.min_foot_height)
        
        return foot

    def _get_swing_state_with_blending(self, leg_name, phase):
        """
        Determine if leg is in swing with smooth blending factor.
        Returns: (in_swing, swing_progress, blend_factor)
        """
        for swing_start, swing_end in self.swing_phases[leg_name]:
            swing_duration = swing_end - swing_start
            
            # Handle phase wraparound for swing intervals crossing phase 1.0
            if swing_end > 1.0:
                swing_end_wrapped = swing_end - 1.0
                if phase >= swing_start or phase < swing_end_wrapped:
                    # Compute wrapped progress
                    if phase >= swing_start:
                        phase_in_swing = phase - swing_start
                    else:
                        phase_in_swing = (1.0 - swing_start) + phase
                    swing_progress = phase_in_swing / swing_duration
                    
                    # Compute blend factor with wider transitions
                    if phase_in_swing < self.transition_window:
                        blend_factor = phase_in_swing / self.transition_window
                        return True, swing_progress, blend_factor
                    elif phase_in_swing > swing_duration - self.transition_window:
                        blend_factor = (swing_duration - phase_in_swing) / self.transition_window
                        return True, swing_progress, blend_factor
                    else:
                        return True, swing_progress, 1.0
            else:
                # Normal case: swing interval within [0, 1)
                if swing_start <= phase < swing_end:
                    swing_progress = (phase - swing_start) / swing_duration
                    phase_in_swing = phase - swing_start
                    
                    # Compute blend factor with wider transitions
                    if phase_in_swing < self.transition_window:
                        blend_factor = phase_in_swing / self.transition_window
                        return True, swing_progress, blend_factor
                    elif phase_in_swing > swing_duration - self.transition_window:
                        blend_factor = (swing_duration - phase_in_swing) / self.transition_window
                        return True, swing_progress, blend_factor
                    else:
                        return True, swing_progress, 1.0
        
        # Not in swing - pure stance
        return False, 0.0, 0.0

    def _compute_continuous_stance_progress(self, leg_name, phase):
        """
        Compute stance progress using continuous modular arithmetic to avoid discontinuities.
        Returns value in [0, 1] representing progress through current stance phase.
        """
        swing_ranges = self.swing_phases[leg_name]
        
        if len(swing_ranges) == 2:
            (sw1_start, sw1_end), (sw2_start, sw2_end) = swing_ranges
            
            # Handle wrapped swing interval
            sw2_end_actual = sw2_end if sw2_end <= 1.0 else sw2_end - 1.0
            
            # Determine which stance interval we're in using continuous logic
            # Stance 1: from sw2_end to sw1_start (may wrap around phase 1.0)
            # Stance 2: from sw1_end to sw2_start
            
            if sw2_end <= 1.0:
                # No wrap in second swing
                if phase >= sw2_end or phase < sw1_start:
                    # In first stance interval (wraps around 1.0)
                    if phase >= sw2_end:
                        phase_in_stance = phase - sw2_end
                    else:
                        phase_in_stance = (1.0 - sw2_end) + phase
                    stance_duration = (1.0 - sw2_end) + sw1_start
                    return np.clip(phase_in_stance / stance_duration, 0.0, 1.0) if stance_duration > 0 else 0.0
                elif sw1_end <= phase < sw2_start:
                    # In second stance interval
                    stance_duration = sw2_start - sw1_end
                    phase_in_stance = phase - sw1_end
                    return np.clip(phase_in_stance / stance_duration, 0.0, 1.0) if stance_duration > 0 else 0.0
            else:
                # Second swing wraps around 1.0
                sw2_end_mod = sw2_end - 1.0
                if sw1_end <= phase < sw2_start:
                    # In stance interval between first and second swing
                    stance_duration = sw2_start - sw1_end
                    phase_in_stance = phase - sw1_end
                    return np.clip(phase_in_stance / stance_duration, 0.0, 1.0) if stance_duration > 0 else 0.0
                else:
                    # In stance after second swing (wraps)
                    if phase >= sw2_end_mod and phase < sw1_start:
                        stance_duration = sw1_start - sw2_end_mod
                        phase_in_stance = phase - sw2_end_mod
                        return np.clip(phase_in_stance / stance_duration, 0.0, 1.0) if stance_duration > 0 else 0.0
        
        # Fallback: simple linear progress
        return np.clip(phase, 0.0, 1.0)

    def _compute_swing_foot_with_velocity_matching(self, foot, swing_progress, stance_progress, leg_name, phase):
        """
        Compute foot position during swing with velocity-matched boundaries.
        """
        # Forward progression during swing centered at midpoint
        foot[0] += self.stance_step_length * (swing_progress - 0.5)
        
        # Quintic vertical lift for smooth acceleration with zero boundary velocities
        foot[2] += self.swing_height * self._quintic_swing_profile(swing_progress)
        
        return foot

    def _compute_stance_foot_with_extension(self, foot, stance_progress, is_left_leg, roll_angle):
        """
        Compute foot position during stance with proper directional extension.
        """
        # Smooth stance progression using quintic for C2 continuity
        stance_progress_smooth = self._quintic_blend(stance_progress)
        
        # Rearward movement during stance
        foot[0] -= self.stance_step_length * stance_progress_smooth
        
        # Directional extension modulation based on roll
        # Positive roll = rolling left, negative roll = rolling right
        # Left legs should extend DOWN (negative Z) when rolling left (positive roll)
        # Right legs should extend DOWN (negative Z) when rolling right (negative roll)
        if is_left_leg:
            # Left leg: extend down when roll is positive (rolling left)
            extension = -roll_angle * self.extension_modulation
        else:
            # Right leg: extend down when roll is negative (rolling right)
            extension = roll_angle * self.extension_modulation
        
        # Apply extension directly without abs() to preserve directionality
        foot[2] += extension
        
        return foot

    def _quintic_blend(self, t):
        """
        Quintic polynomial blending function for C2 continuity.
        t: parameter from 0 to 1
        Returns: smoothed value from 0 to 1 with zero first and second derivatives at boundaries
        """
        t = np.clip(t, 0.0, 1.0)
        return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3

    def _quintic_swing_profile(self, s):
        """
        Quintic polynomial for smooth swing trajectory with zero velocity/accel at boundaries.
        """
        s = np.clip(s, 0.0, 1.0)
        return 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3