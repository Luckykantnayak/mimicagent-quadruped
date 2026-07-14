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
        
        # Takeoff parameters
        self.takeoff_vz = 2.5  # Vertical velocity during takeoff
        self.gravity = 9.81
        
        # Yaw injection parameters
        self.yaw_rate_max = 3.0  # Max yaw rate during injection (rad/s)
        self.target_yaw_change = np.pi / 2  # 90 degree rotation
        
        # Limb motion parameters
        self.retract_distance = 0.15  # How far limbs pull inward
        self.sweep_radius = 0.2  # Radius of circular sweep motion
        self.landing_extension = 0.1  # How far legs extend for landing
        
        # Diagonal pairing for yaw generation
        # FL and RR sweep forward, FR and RL sweep backward
        self.sweep_direction = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.sweep_direction[leg] = 1.0  # Forward sweep
            else:
                self.sweep_direction[leg] = -1.0  # Backward sweep

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on current phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        if phase < 0.2:
            # Takeoff phase: accelerate upward
            progress = phase / 0.2
            vz = self.takeoff_vz * progress
            
        elif phase < 0.35:
            # Liftoff and retraction: ballistic flight
            time_since_takeoff = (phase - 0.2) / 0.15 / self.freq
            vz = self.takeoff_vz - self.gravity * time_since_takeoff
            
        elif phase < 0.6:
            # Yaw injection: ramp up then sustain yaw rate
            progress = (phase - 0.35) / 0.25
            time_since_takeoff = (phase - 0.2) / self.freq
            vz = self.takeoff_vz - self.gravity * time_since_takeoff
            
            # Smooth ramp-up of yaw rate using smoothstep
            smoothstep = 3 * progress**2 - 2 * progress**3
            yaw_rate = self.yaw_rate_max * smoothstep
            
        elif phase < 0.8:
            # Yaw damping: ramp down yaw rate
            progress = (phase - 0.6) / 0.2
            time_since_takeoff = (phase - 0.2) / self.freq
            vz = self.takeoff_vz - self.gravity * time_since_takeoff
            
            # Smooth ramp-down of yaw rate
            smoothstep = 3 * progress**2 - 2 * progress**3
            yaw_rate = self.yaw_rate_max * (1.0 - smoothstep)
            
        else:
            # Landing phase: decelerate downward motion
            progress = (phase - 0.8) / 0.2
            time_since_takeoff = (phase - 0.2) / self.freq
            vz_ballistic = self.takeoff_vz - self.gravity * time_since_takeoff
            
            # Smooth deceleration to zero at landing
            vz = vz_ballistic * (1.0 - progress)
        
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
            # Takeoff phase: extend downward
            progress = phase / 0.2
            foot[2] = base_z - self.landing_extension * progress
            
        elif phase < 0.35:
            # Retraction phase: pull inward and upward toward COM
            progress = (phase - 0.2) / 0.15
            smoothstep = 3 * progress**2 - 2 * progress**3
            
            # Move foot toward body center (0, 0, 0)
            foot[0] = base_x * (1.0 - smoothstep * 0.5)
            foot[1] = base_y * (1.0 - smoothstep * 0.5)
            foot[2] = base_z - self.landing_extension + smoothstep * (self.landing_extension + self.retract_distance)
            
        elif phase < 0.6:
            # Yaw injection: asymmetric circular sweep in XY plane
            progress = (phase - 0.35) / 0.25
            
            # Retracted base position
            retract_x = base_x * 0.5
            retract_y = base_y * 0.5
            retract_z = base_z + self.retract_distance
            
            # Circular sweep motion in horizontal plane
            sweep_angle = 2 * np.pi * progress * self.sweep_direction[leg_name]
            sweep_x = self.sweep_radius * np.cos(sweep_angle)
            sweep_y = self.sweep_radius * np.sin(sweep_angle)
            
            foot[0] = retract_x + sweep_x * 0.3
            foot[1] = retract_y + sweep_y * 0.3
            foot[2] = retract_z
            
        elif phase < 0.8:
            # Yaw damping: reverse sweep motion
            progress = (phase - 0.6) / 0.2
            
            # Retracted base position
            retract_x = base_x * 0.5
            retract_y = base_y * 0.5
            retract_z = base_z + self.retract_distance
            
            # Reverse sweep with deceleration
            sweep_angle_start = 2 * np.pi * self.sweep_direction[leg_name]
            sweep_angle = sweep_angle_start * (1.0 - progress)
            sweep_x = self.sweep_radius * np.cos(sweep_angle)
            sweep_y = self.sweep_radius * np.sin(sweep_angle)
            
            smoothstep = 3 * progress**2 - 2 * progress**3
            
            foot[0] = retract_x + sweep_x * 0.3 * (1.0 - smoothstep)
            foot[1] = retract_y + sweep_y * 0.3 * (1.0 - smoothstep)
            foot[2] = retract_z
            
        else:
            # Landing phase: extend downward and absorb impact
            progress = (phase - 0.8) / 0.2
            smoothstep = 3 * progress**2 - 2 * progress**3
            
            # Start from retracted position, extend to landing
            retract_x = base_x * 0.5
            retract_y = base_y * 0.5
            retract_z = base_z + self.retract_distance
            
            # Extend to landing position
            foot[0] = retract_x + (base_x - retract_x) * smoothstep
            foot[1] = retract_y + (base_y - retract_y) * smoothstep
            foot[2] = retract_z - (retract_z - base_z + self.landing_extension) * smoothstep
            
            # Impact absorption: slight additional flexion after contact
            if progress > 0.5:
                absorption_progress = (progress - 0.5) / 0.5
                foot[2] -= 0.05 * np.sin(np.pi * absorption_progress)
        
        return foot