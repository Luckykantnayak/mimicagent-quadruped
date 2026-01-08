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
        self.orbit_angular_velocity = 2.0 * np.pi * self.freq  # rad/s for one orbit per cycle
        self.forward_speed = 0.6
        
        # Roll oscillation parameters
        self.roll_amplitude = 0.35  # radians (~20 degrees)
        self.roll_frequency = 2.0 * self.freq  # Two complete roll cycles per orbit
        
        # Leg motion parameters
        self.swing_duration = 0.125  # Fraction of phase cycle
        self.swing_height = 0.06
        self.stance_step_length = 0.15
        self.extension_modulation = 0.08  # Vertical extension range for tilt support
        
        # Phase offsets for diagonal coordination
        # Group 1: FL, RR swing together
        # Group 2: FR, RL swing together
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
        
        Circular motion: vx, vy modulated to trace circle, constant yaw rate
        Roll motion: sinusoidal roll rate creates oscillating tilt
        """
        # Circular orbit velocities
        # Phase 0: start of circle, moving forward
        # Use sinusoidal modulation for smooth circular path
        orbit_angle = 2.0 * np.pi * phase
        
        vx = self.forward_speed * np.cos(orbit_angle)
        vy = self.forward_speed * np.sin(orbit_angle)
        vz = 0.0
        
        # Constant yaw rate for circular turning
        yaw_rate = self.orbit_angular_velocity
        
        # Roll rate: two complete oscillations per orbit cycle
        # Peaks: phase 0.25 (left), 0.5 (right), 0.75 (left), 1.0 (neutral)
        # Roll angle = amplitude * sin(4π * phase)
        # Roll rate = d(roll)/dt = amplitude * 4π * freq * cos(4π * phase)
        roll_angle_phase = 4.0 * np.pi * phase
        roll_rate = -self.roll_amplitude * 4.0 * np.pi * self.freq * np.cos(roll_angle_phase)
        
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

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with:
        - Swing phases for diagonal pairs at appropriate times
        - Stance phase with rearward progression
        - Vertical extension modulation based on roll direction
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Check if leg is in swing phase
        in_swing = False
        swing_progress = 0.0
        
        for swing_start, swing_end in self.swing_phases[leg_name]:
            if swing_start <= phase < swing_end:
                in_swing = True
                swing_progress = (phase - swing_start) / (swing_end - swing_start)
                break
        
        # Compute current roll angle for extension modulation
        roll_angle = self.roll_amplitude * np.sin(4.0 * np.pi * phase)
        
        # Determine if leg is on left or right side
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        if in_swing:
            # Swing phase: arc trajectory
            # Move foot forward and lift it
            swing_angle = np.pi * swing_progress
            
            # Forward progression during swing
            foot[0] += self.stance_step_length * (swing_progress - 0.5)
            
            # Vertical lift
            foot[2] += self.swing_height * np.sin(swing_angle)
            
        else:
            # Stance phase: foot moves rearward relative to body (body advances)
            # Compute stance progress within current stance interval
            stance_progress = self._compute_stance_progress(leg_name, phase)
            
            # Rearward movement during stance
            foot[0] -= self.stance_step_length * stance_progress
            
            # Vertical extension modulation based on roll
            # Outer legs (opposite side of roll) extend more
            # Inner legs (same side as roll) compress more
            if is_left_leg:
                # Left leg: extend when rolling right (negative roll), compress when rolling left (positive roll)
                extension = -roll_angle * self.extension_modulation
            else:
                # Right leg: extend when rolling left (positive roll), compress when rolling right (negative roll)
                extension = roll_angle * self.extension_modulation
            
            foot[2] += extension
        
        return foot
    
    def _compute_stance_progress(self, leg_name, phase):
        """
        Compute progress through current stance phase [0, 1].
        """
        # Find the current stance interval
        swing_ranges = self.swing_phases[leg_name]
        
        # Build stance intervals (complement of swing intervals)
        if len(swing_ranges) == 2:
            # Stance intervals between and around swings
            (sw1_start, sw1_end), (sw2_start, sw2_end) = swing_ranges
            
            # Three stance intervals:
            # [0, sw1_start), [sw1_end, sw2_start), [sw2_end, 1.0)
            if phase < sw1_start:
                stance_duration = sw1_start
                stance_start = 0.0
            elif sw1_end <= phase < sw2_start:
                stance_duration = sw2_start - sw1_end
                stance_start = sw1_end
            elif phase >= sw2_end:
                stance_duration = 1.0 - sw2_end
                stance_start = sw2_end
            else:
                # Should not reach here if phase is in stance
                return 0.0
            
            if stance_duration > 0:
                return (phase - stance_start) / stance_duration
            else:
                return 0.0
        else:
            # Fallback: simple linear progress
            return phase