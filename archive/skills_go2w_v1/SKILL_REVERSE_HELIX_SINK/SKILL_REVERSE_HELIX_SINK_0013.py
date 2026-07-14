from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_REVERSE_HELIX_SINK_MotionGenerator(BaseMotionGenerator):
    """
    Reverse helix sink: backward translation, counter-clockwise yaw rotation, and vertical descent.
    
    Motion characteristics:
    - Base moves backward continuously (negative vx in world frame)
    - Base rotates counter-clockwise continuously (positive yaw rate)
    - Base descends gradually (negative vz)
    - All four legs remain in contact throughout, adjusting body-frame positions
    - One complete phase cycle = one full helical loop (360° rotation)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Lower frequency for smooth, controlled helical descent
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.backward_velocity = -0.3  # Backward motion (negative x in world frame)
        self.yaw_rate = 2 * np.pi * self.freq  # 360° rotation per phase cycle
        self.descent_rate = -0.15  # Gradual descent rate
        
        # Height and spread parameters
        self.max_descent = 0.25  # Maximum descent distance
        self.max_lateral_spread = 0.05  # Leg spreading for stability at low height
        self.max_longitudinal_adjust = 0.04  # Forward/backward adjustment for yaw
        
        # Phase-dependent descent profile (normalized)
        # Accelerates in first half, continues in second half
        self.descent_profile_phases = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.descent_profile_values = [0.0, 0.2, 0.5, 0.85, 1.0]  # Cumulative descent fraction

    def update_base_motion(self, phase, dt):
        """
        Update base with backward velocity, counter-clockwise yaw, and descent.
        
        Phase 0.0-0.25: Initiation and acceleration
        Phase 0.25-0.5: Peak descent and rotation
        Phase 0.5-0.75: Sustained spiral descent
        Phase 0.75-1.0: Completion and reset
        """
        
        # Backward velocity (world frame, negative x)
        vx = self.backward_velocity
        
        # No lateral drift
        vy = 0.0
        
        # Vertical descent - varies by phase for smooth acceleration/deceleration
        if phase < 0.25:
            # Ramp up descent
            descent_factor = phase / 0.25
        elif phase < 0.5:
            # Peak descent
            descent_factor = 1.0
        elif phase < 0.75:
            # Sustained descent
            descent_factor = 0.8
        else:
            # Taper descent toward end
            descent_factor = 0.5 * (1.0 - (phase - 0.75) / 0.25)
        
        vz = self.descent_rate * descent_factor
        
        # Counter-clockwise yaw rate (positive, constant for smooth 360° rotation)
        yaw_rate = self.yaw_rate
        
        # Set velocities
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
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
        Compute foot position in body frame with continuous stance adjustment.
        
        All legs remain in contact. Body-frame positions adjust to:
        1. Accommodate yaw rotation (feet rotate relative to body)
        2. Lower base height (legs compress)
        3. Spread laterally for stability at low height
        4. Adjust longitudinally for backward motion stability
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Get cumulative descent fraction using interpolation
        descent_fraction = np.interp(phase, self.descent_profile_phases, self.descent_profile_values)
        
        # Vertical compression: raise foot in body frame as base descends
        # (negative z in body frame = foot moves up relative to body = base moves down)
        foot[2] += self.max_descent * descent_fraction
        
        # Lateral spreading: increase stance width for stability as height decreases
        lateral_spread = self.max_lateral_spread * descent_fraction
        
        if leg_name.startswith('FL') or leg_name.startswith('RL'):
            # Left legs: spread further left (positive y in body frame)
            foot[1] += lateral_spread
        else:
            # Right legs: spread further right (negative y in body frame)
            foot[1] -= lateral_spread
        
        # Longitudinal adjustment for yaw-induced stability shift
        # Front legs move slightly forward, rear legs slightly back
        longitudinal_adjust = self.max_longitudinal_adjust * descent_fraction
        
        if leg_name.startswith('FL') or leg_name.startswith('FR'):
            # Front legs: extend forward
            foot[0] += longitudinal_adjust
        else:
            # Rear legs: extend backward
            foot[0] -= longitudinal_adjust
        
        # Additional phase-dependent rotational compensation
        # As base yaws, feet need slight cyclic adjustment to maintain ground contact
        # This simulates the body-frame shift due to continuous rotation
        yaw_phase_offset = 2 * np.pi * phase  # Current yaw angle in radians
        
        # Small cyclic x-y adjustment to simulate rotational stability maintenance
        cyclic_amplitude = 0.02 * descent_fraction  # Increases with descent
        
        if leg_name.startswith('FL'):
            foot[0] += cyclic_amplitude * np.cos(yaw_phase_offset)
            foot[1] += cyclic_amplitude * np.sin(yaw_phase_offset)
        elif leg_name.startswith('FR'):
            foot[0] += cyclic_amplitude * np.cos(yaw_phase_offset + np.pi/2)
            foot[1] += cyclic_amplitude * np.sin(yaw_phase_offset + np.pi/2)
        elif leg_name.startswith('RL'):
            foot[0] += cyclic_amplitude * np.cos(yaw_phase_offset + np.pi)
            foot[1] += cyclic_amplitude * np.sin(yaw_phase_offset + np.pi)
        elif leg_name.startswith('RR'):
            foot[0] += cyclic_amplitude * np.cos(yaw_phase_offset + 3*np.pi/2)
            foot[1] += cyclic_amplitude * np.sin(yaw_phase_offset + 3*np.pi/2)
        
        return foot