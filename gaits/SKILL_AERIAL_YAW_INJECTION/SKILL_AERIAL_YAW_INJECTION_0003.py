from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_AERIAL_YAW_INJECTION_MotionGenerator(BaseMotionGenerator):
    """
    Aerial yaw injection maneuver: vertical takeoff, mid-air yaw reorientation
    via asymmetric limb momentum exchange, and controlled landing.
    
    Phase breakdown:
      [0.0, 0.2]  : Takeoff - symmetric vertical push
      [0.2, 0.35] : Liftoff and retraction - legs pull inward
      [0.35, 0.6] : Yaw injection - asymmetric limb sweeps induce body yaw
      [0.6, 0.8]  : Yaw damping - reverse sweeps to arrest rotation
      [0.8, 1.0]  : Landing - symmetric extension and impact absorption
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower cycle to allow sufficient flight time
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Trajectory parameters - reduced for safety
        self.peak_height = 0.28  # Reduced from 0.35m to stay within envelope
        self.yaw_rate_max = 2.5  # Max yaw rate during injection (rad/s)
        
        # Limb motion parameters - reduced to avoid joint limits
        self.retract_distance = 0.07  # Reduced from 0.12m - less aggressive vertical retraction
        self.retract_xy_factor = 0.82  # Increased from 0.7 - less aggressive lateral retraction
        self.sweep_amplitude = 0.045  # Reduced from 0.08m - smaller sweep for joint safety
        self.landing_extension = 0.09  # Increased from 0.08m to match retraction better
        
        # Diagonal pairing for yaw generation
        self.sweep_direction = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.sweep_direction[leg] = 1.0  # Forward sweep
            else:
                self.sweep_direction[leg] = -1.0  # Backward sweep

    def _compute_vertical_trajectory(self, phase):
        """
        Compute vertical position and velocity using smooth kinematic trajectory.
        Returns absolute position (not relative) and velocity to ensure proper integration.
        """
        if phase < 0.2:
            # Takeoff: smooth acceleration upward
            progress = phase / 0.2
            smoothstep = 3 * progress**2 - 2 * progress**3
            z_pos = 0.08 * smoothstep
            # Velocity: derivative of position with respect to time
            z_vel = (0.08 * 6 * (progress - progress**2)) / 0.2 * self.freq
            
        elif phase < 0.8:
            # Aerial phase: sinusoidal arc from takeoff to landing prep
            # Map phase [0.2, 0.8] to arc [0, pi]
            arc_progress = (phase - 0.2) / 0.6
            arc_angle = arc_progress * np.pi
            
            # Sinusoidal altitude profile
            z_pos = 0.08 + (self.peak_height - 0.08) * np.sin(arc_angle)
            
            # Velocity derivative
            z_vel = (self.peak_height - 0.08) * np.cos(arc_angle) * np.pi / 0.6 * self.freq
            
        else:
            # Landing: smooth deceleration to ground
            progress = (phase - 0.8) / 0.2
            smoothstep = 3 * progress**2 - 2 * progress**3
            
            # At phase 0.8, z is at end of aerial arc: sin(pi) = 0
            z_start = 0.08 + (self.peak_height - 0.08) * np.sin(np.pi)  # = 0.08
            z_pos = z_start * (1.0 - smoothstep)
            
            # Velocity derivative
            z_vel = -z_start * 6 * (progress - progress**2) / 0.2 * self.freq
            
            # Impact absorption after phase 0.9 - add small compression
            if progress > 0.5:
                absorption_progress = (progress - 0.5) / 0.5
                absorption_depth = 0.02  # Reduced absorption depth
                z_pos -= absorption_depth * (1.0 - np.cos(np.pi * absorption_progress))
                z_vel -= absorption_depth * np.pi * np.sin(np.pi * absorption_progress) / 0.1 * self.freq
        
        return z_pos, z_vel

    def _compute_yaw_trajectory(self, phase):
        """
        Compute yaw angular velocity using smooth phase-based profile.
        """
        if phase < 0.35:
            return 0.0
            
        elif phase < 0.6:
            # Yaw injection: smooth ramp up and sustain
            progress = (phase - 0.35) / 0.25
            # Smoothstep for gradual acceleration
            smoothstep = 3 * progress**2 - 2 * progress**3
            return self.yaw_rate_max * smoothstep
            
        elif phase < 0.8:
            # Yaw damping: smooth ramp down
            progress = (phase - 0.6) / 0.2
            smoothstep = 3 * progress**2 - 2 * progress**3
            return self.yaw_rate_max * (1.0 - smoothstep)
            
        else:
            return 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on current phase.
        """
        # Compute vertical trajectory
        z_pos, z_vel = self._compute_vertical_trajectory(phase)
        
        # Compute yaw rate
        yaw_rate = self._compute_yaw_trajectory(phase)
        
        # No horizontal translation
        vx = 0.0
        vy = 0.0
        vz = z_vel
        roll_rate = 0.0
        pitch_rate = 0.0
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase and leg coordination.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        base_x = foot[0]
        base_y = foot[1]
        base_z = foot[2]
        
        if phase < 0.2:
            # Takeoff phase: extend downward for push
            progress = phase / 0.2
            smoothstep = 3 * progress**2 - 2 * progress**3
            foot[2] = base_z - self.landing_extension * smoothstep
            
        elif phase < 0.35:
            # Retraction phase: pull inward and upward toward COM
            progress = (phase - 0.2) / 0.15
            smoothstep = 3 * progress**2 - 2 * progress**3
            
            # Start from extended position, retract toward body
            start_z = base_z - self.landing_extension
            target_z = base_z + self.retract_distance
            
            foot[0] = base_x * (1.0 - (1.0 - self.retract_xy_factor) * smoothstep)
            foot[1] = base_y * (1.0 - (1.0 - self.retract_xy_factor) * smoothstep)
            foot[2] = start_z + (target_z - start_z) * smoothstep
            
        elif phase < 0.6:
            # Yaw injection: asymmetric circular sweep in XY plane
            progress = (phase - 0.35) / 0.25
            
            # Smooth amplitude envelope: ramp up and down for smoother transitions
            amplitude_envelope = np.sin(np.pi * progress)
            
            # Retracted base position
            retract_x = base_x * self.retract_xy_factor
            retract_y = base_y * self.retract_xy_factor
            retract_z = base_z + self.retract_distance
            
            # Smooth circular sweep motion
            sweep_angle = np.pi * progress * self.sweep_direction[leg_name]
            sweep_x = self.sweep_amplitude * amplitude_envelope * np.sin(sweep_angle)
            sweep_y = self.sweep_amplitude * amplitude_envelope * (1.0 - np.cos(sweep_angle)) * np.sign(base_y)
            
            foot[0] = retract_x + sweep_x
            foot[1] = retract_y + sweep_y
            foot[2] = retract_z
            
        elif phase < 0.8:
            # Yaw damping: reverse sweep motion smoothly
            progress = (phase - 0.6) / 0.2
            smoothstep = 3 * progress**2 - 2 * progress**3
            
            # Retracted base position
            retract_x = base_x * self.retract_xy_factor
            retract_y = base_y * self.retract_xy_factor
            retract_z = base_z + self.retract_distance
            
            # Decay sweep back to retracted position with smooth envelope
            sweep_decay = 1.0 - smoothstep
            amplitude_envelope = np.sin(np.pi * (1.0 - progress))
            sweep_angle = np.pi * self.sweep_direction[leg_name] * sweep_decay
            sweep_x = self.sweep_amplitude * amplitude_envelope * np.sin(sweep_angle)
            sweep_y = self.sweep_amplitude * amplitude_envelope * (1.0 - np.cos(sweep_angle)) * np.sign(base_y)
            
            foot[0] = retract_x + sweep_x
            foot[1] = retract_y + sweep_y
            foot[2] = retract_z
            
        else:
            # Landing phase: extend downward and outward to landing position
            progress = (phase - 0.8) / 0.2
            smoothstep = 3 * progress**2 - 2 * progress**3
            
            # Start from retracted position
            start_x = base_x * self.retract_xy_factor
            start_y = base_y * self.retract_xy_factor
            start_z = base_z + self.retract_distance
            
            # Target is extended landing position
            target_z = base_z - self.landing_extension
            
            foot[0] = start_x + (base_x - start_x) * smoothstep
            foot[1] = start_y + (base_y - start_y) * smoothstep
            foot[2] = start_z + (target_z - start_z) * smoothstep
            
            # Impact absorption: slight additional compression after initial contact
            if progress > 0.6:
                absorption_progress = (progress - 0.6) / 0.4
                foot[2] -= 0.02 * np.sin(np.pi * absorption_progress)
        
        return foot