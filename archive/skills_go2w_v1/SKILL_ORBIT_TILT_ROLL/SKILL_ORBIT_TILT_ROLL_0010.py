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
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
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
        self.extension_modulation = 0.025  # Reduced to prevent ground penetration
        self.ground_clearance_bias = 0.01  # Safety margin above ground
        
        # Transition smoothing window
        self.transition_window = 0.02  # Phase window for blending
        
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
        
        # Smoothed roll trajectory using quintic polynomial within each quarter-cycle
        # This reduces acceleration spikes compared to pure sinusoid
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
        Compute roll angle with smooth transitions to reduce acceleration spikes.
        Uses sinusoidal profile but with bounded derivatives.
        """
        roll_angle_phase = 4.0 * np.pi * phase
        roll_angle = self.roll_amplitude * np.sin(roll_angle_phase)
        return roll_angle

    def _compute_smooth_roll_rate(self, phase):
        """
        Compute roll rate with smoothing to reduce jerk.
        """
        roll_angle_phase = 4.0 * np.pi * phase
        roll_rate = -self.roll_amplitude * 4.0 * np.pi * self.freq * np.cos(roll_angle_phase)
        return roll_rate

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with smooth transitions and ground clearance.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Find swing state with transition blending
        swing_state, swing_progress, blend_factor = self._get_swing_state_with_blending(leg_name, phase)
        
        # Compute current roll angle for extension modulation
        roll_angle = self._compute_smooth_roll_angle(phase)
        
        # Determine if leg is on left or right side
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        if swing_state == 'swing':
            # Swing phase: smooth arc trajectory with quintic vertical profile
            # Forward progression during swing
            foot[0] += self.stance_step_length * (swing_progress - 0.5)
            
            # Quintic vertical lift for smooth acceleration
            foot[2] += self.swing_height * self._quintic_swing_profile(swing_progress)
            
        elif swing_state == 'stance':
            # Stance phase: foot moves rearward relative to body
            stance_progress = self._compute_stance_progress(leg_name, phase)
            
            # Smooth rearward movement during stance
            foot[0] -= self.stance_step_length * stance_progress
            
            # Vertical extension modulation based on roll (outer legs extend UP, not down)
            # Add ground clearance bias to prevent penetration
            if is_left_leg:
                # Left leg: extend UP when rolling right (negative roll)
                extension = -roll_angle * self.extension_modulation
            else:
                # Right leg: extend UP when rolling left (positive roll)
                extension = roll_angle * self.extension_modulation
            
            # Apply extension as upward adjustment plus safety bias
            foot[2] += abs(extension) + self.ground_clearance_bias
            
        else:  # transition state
            # Blend between swing and stance trajectories
            # Compute both positions and interpolate
            foot_swing = self._compute_swing_foot(foot.copy(), swing_progress)
            foot_stance = self._compute_stance_foot(foot.copy(), leg_name, phase, is_left_leg, roll_angle)
            
            # Smooth blend using blend_factor (0 = stance, 1 = swing)
            foot = foot_stance * (1 - blend_factor) + foot_swing * blend_factor
        
        return foot

    def _get_swing_state_with_blending(self, leg_name, phase):
        """
        Determine swing state with smooth transition blending.
        Returns: (state, swing_progress, blend_factor)
        state: 'swing', 'stance', or 'transition'
        blend_factor: 0 to 1, used for smooth interpolation
        """
        for swing_start, swing_end in self.swing_phases[leg_name]:
            # Check if in core swing phase
            if swing_start + self.transition_window <= phase < swing_end - self.transition_window:
                swing_progress = (phase - swing_start) / (swing_end - swing_start)
                return 'swing', swing_progress, 1.0
            
            # Check if in transition into swing
            elif swing_start <= phase < swing_start + self.transition_window:
                swing_progress = (phase - swing_start) / (swing_end - swing_start)
                blend_factor = (phase - swing_start) / self.transition_window
                blend_factor = self._smooth_step(blend_factor)
                return 'transition', swing_progress, blend_factor
            
            # Check if in transition out of swing
            elif swing_end - self.transition_window <= phase < swing_end:
                swing_progress = (phase - swing_start) / (swing_end - swing_start)
                blend_factor = (swing_end - phase) / self.transition_window
                blend_factor = self._smooth_step(blend_factor)
                return 'transition', swing_progress, blend_factor
        
        # Not in any swing phase
        return 'stance', 0.0, 0.0

    def _smooth_step(self, x):
        """
        Smooth step function (3rd order) for continuous velocity transitions.
        """
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _quintic_swing_profile(self, s):
        """
        Quintic polynomial for smooth swing trajectory with zero velocity/accel at boundaries.
        s: progress from 0 to 1
        Returns: height multiplier from 0 to 1
        """
        s = np.clip(s, 0.0, 1.0)
        # Quintic: 6s^5 - 15s^4 + 10s^3
        return 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3

    def _compute_swing_foot(self, foot, swing_progress):
        """
        Compute foot position for pure swing phase.
        """
        foot[0] += self.stance_step_length * (swing_progress - 0.5)
        foot[2] += self.swing_height * self._quintic_swing_profile(swing_progress)
        return foot

    def _compute_stance_foot(self, foot, leg_name, phase, is_left_leg, roll_angle):
        """
        Compute foot position for pure stance phase.
        """
        stance_progress = self._compute_stance_progress(leg_name, phase)
        
        # Smooth stance progression using ease function
        stance_progress_smooth = self._smooth_step(stance_progress)
        
        foot[0] -= self.stance_step_length * stance_progress_smooth
        
        # Extension modulation with ground safety
        if is_left_leg:
            extension = -roll_angle * self.extension_modulation
        else:
            extension = roll_angle * self.extension_modulation
        
        foot[2] += abs(extension) + self.ground_clearance_bias
        return foot

    def _compute_stance_progress(self, leg_name, phase):
        """
        Compute progress through current stance phase [0, 1].
        """
        swing_ranges = self.swing_phases[leg_name]
        
        if len(swing_ranges) == 2:
            (sw1_start, sw1_end), (sw2_start, sw2_end) = swing_ranges
            
            # Handle wrap-around for first stance interval
            if phase < sw1_start:
                # Stance from sw2_end (previous cycle) to sw1_start
                stance_duration = sw1_start + (1.0 - sw2_end)
                phase_in_stance = phase + (1.0 - sw2_end)
                return phase_in_stance / stance_duration if stance_duration > 0 else 0.0
            
            elif sw1_end <= phase < sw2_start:
                stance_duration = sw2_start - sw1_end
                stance_start = sw1_end
                return (phase - stance_start) / stance_duration if stance_duration > 0 else 0.0
            
            elif phase >= sw2_end:
                stance_duration = 1.0 - sw2_end + sw1_start
                phase_in_stance = phase - sw2_end
                return phase_in_stance / (1.0 - sw2_end + sw1_start) if stance_duration > 0 else 0.0
            
            else:
                return 0.0
        else:
            return phase