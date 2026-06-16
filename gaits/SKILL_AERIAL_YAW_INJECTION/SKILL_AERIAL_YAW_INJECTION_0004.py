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
        self.freq = 0.5
        
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Reduced trajectory parameters to stay within safe envelope
        self.peak_height = 0.14  # Reduced from 0.28m to keep base height within limits
        self.yaw_rate_max = 6.5
        
        # Reduced limb motion parameters for joint safety
        self.retract_distance = 0.1  # Reduced from 0.07m to minimize workspace compression
        self.retract_xy_factor = 0.7  # Increased from 0.82 to reduce lateral retraction
        self.sweep_amplitude = 0.08  # Reduced from 0.045m to avoid joint limits
        self.landing_extension = 0.045  # Matched to retract_distance for symmetry
        
        # Diagonal pairing for yaw generation
        self.sweep_direction = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.sweep_direction[leg] = 1.0
            else:
                self.sweep_direction[leg] = -1.0

    def _compute_vertical_trajectory(self, phase):
        """
        Compute vertical velocity using smooth kinematic trajectory.
        Works in relative displacement to maintain proper base height envelope.
        """
        if phase < 0.2:
            # Takeoff: smooth acceleration upward
            progress = phase / 0.2
            smoothstep = 3 * progress**2 - 2 * progress**3
            # Target displacement during takeoff
            takeoff_height = 0.05
            # Velocity derivative
            z_vel = (takeoff_height * 6 * (progress - progress**2)) / 0.2 * self.freq
            
        elif phase < 0.8:
            # Aerial phase: sinusoidal arc trajectory
            arc_progress = (phase - 0.2) / 0.6
            arc_angle = arc_progress * np.pi
            
            # Velocity follows derivative of sin arc from takeoff to landing
            z_vel = self.peak_height * np.cos(arc_angle) * np.pi / 0.6 * self.freq
            
        else:
            # Landing: smooth deceleration to ground
            progress = (phase - 0.8) / 0.2
            # Descent velocity component
            landing_height = 0.05
            z_vel = -landing_height * 6 * (progress - progress**2) / 0.2 * self.freq
            
            # Gentle impact absorption after midpoint
            if progress > 0.5:
                absorption_progress = (progress - 0.5) / 0.5
                absorption_depth = 0.008  # Minimal absorption to avoid envelope violation
                z_vel -= absorption_depth * np.pi * np.sin(np.pi * absorption_progress) / 0.1 * self.freq
        
        return z_vel

    def _compute_yaw_trajectory(self, phase):
        """
        Compute yaw angular velocity using smooth phase-based profile.
        """
        if phase < 0.35:
            return 0.0
            
        elif phase < 0.6:
            # Yaw injection: smooth ramp up
            progress = (phase - 0.35) / 0.25
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
        vz = self._compute_vertical_trajectory(phase)
        yaw_rate = self._compute_yaw_trajectory(phase)
        
        vx = 0.0
        vy = 0.0
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
            
            start_z = base_z - self.landing_extension
            target_z = base_z + self.retract_distance
            
            foot[0] = base_x * (1.0 - (1.0 - self.retract_xy_factor) * smoothstep)
            foot[1] = base_y * (1.0 - (1.0 - self.retract_xy_factor) * smoothstep)
            foot[2] = start_z + (target_z - start_z) * smoothstep
            
        elif phase < 0.6:
            # Yaw injection: asymmetric circular sweep with conservative envelope
            progress = (phase - 0.35) / 0.25
            
            # Reduced peak amplitude envelope to avoid joint limits
            amplitude_envelope = 0.7 * np.sin(np.pi * progress)
            
            retract_x = base_x * self.retract_xy_factor
            retract_y = base_y * self.retract_xy_factor
            retract_z = base_z + self.retract_distance
            
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
            
            retract_x = base_x * self.retract_xy_factor
            retract_y = base_y * self.retract_xy_factor
            retract_z = base_z + self.retract_distance
            
            sweep_decay = 1.0 - smoothstep
            amplitude_envelope = 0.7 * np.sin(np.pi * (1.0 - progress))
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
            
            start_x = base_x * self.retract_xy_factor
            start_y = base_y * self.retract_xy_factor
            start_z = base_z + self.retract_distance
            
            target_z = base_z - self.landing_extension
            
            foot[0] = start_x + (base_x - start_x) * smoothstep
            foot[1] = start_y + (base_y - start_y) * smoothstep
            foot[2] = start_z + (target_z - start_z) * smoothstep
            
            # Minimal impact absorption to avoid excessive extension
            if progress > 0.6:
                absorption_progress = (progress - 0.6) / 0.4
                foot[2] -= 0.008 * np.sin(np.pi * absorption_progress)
        
        return foot