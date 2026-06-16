from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_AUGER_DRILL_DESCENT_MotionGenerator(BaseMotionGenerator):
    """
    Auger drill descent motion: continuous vertical descent with rapid yaw rotation.
    
    - Base commands constant downward velocity (vz < 0) and constant positive yaw rate
    - All four legs maintain ground contact throughout entire phase cycle
    - In body frame: feet orbit circularly around vertical axis due to body rotation
    - In world frame: feet remain approximately planted while body rotates and descends
    - Creates helical drilling visual with wide drill-bit radius from extended legs
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0  # One complete drill cycle per second
        
        # Base foot positions
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Angular offsets for symmetric four-fold distribution around body vertical axis
        # FL at 0°, FR at 90°, RL at 180°, RR at 270°
        self.angular_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL'):
                self.angular_offsets[leg] = 0.0
            elif leg.startswith('FR'):
                self.angular_offsets[leg] = np.pi / 2.0
            elif leg.startswith('RL'):
                self.angular_offsets[leg] = np.pi
            elif leg.startswith('RR'):
                self.angular_offsets[leg] = 3.0 * np.pi / 2.0
        
        # Drill motion parameters
        self.descent_rate = -0.12  # Constant downward velocity (m/s) - reduced for smoother motion
        self.yaw_rate = 4.0 * np.pi  # Constant yaw rotation rate (rad/s) - 2 full rotations per cycle
        
        # Leg extension parameters for drill-bit radius
        # Compute radial distance from body center for each foot
        self.foot_radii = {}
        self.foot_z_offsets = {}
        for leg in leg_names:
            base_pos = self.base_feet_pos_body[leg]
            self.foot_radii[leg] = np.sqrt(base_pos[0]**2 + base_pos[1]**2)
            # Start with ground-level z offset (raised to ensure no initial penetration)
            self.foot_z_offsets[leg] = base_pos[2] + 0.05  # Raise by 5cm to ensure ground clearance
        
        # Scale up radial extension for wider drill-bit visual (reduced from 1.2x to 1.05x for kinematic safety)
        for leg in leg_names:
            self.foot_radii[leg] *= 1.05
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track accumulated yaw angle and vertical descent for body-frame foot position computation
        self.accumulated_yaw = 0.0
        self.accumulated_descent = 0.0  # Track total vertical displacement to compensate in body frame

    def update_base_motion(self, phase, dt):
        """
        Update base with constant downward velocity and constant yaw rotation.
        This creates helical descent path in world frame.
        """
        # Constant downward linear velocity in world frame
        self.vel_world = np.array([0.0, 0.0, self.descent_rate])
        
        # Constant positive yaw angular velocity in world frame
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Update accumulated yaw for body-frame foot trajectory computation
        self.accumulated_yaw += self.yaw_rate * dt
        
        # Update accumulated descent (negative because descending)
        self.accumulated_descent += self.descent_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame as circular orbit around vertical axis.
        
        As body rotates, feet appear to orbit in body frame while staying planted in world.
        Each leg maintains constant radial distance and angular offset relative to others.
        Vertical component compensates for base descent to maintain world-frame ground contact.
        """
        # Current angular position in body frame based on accumulated yaw
        # Subtract accumulated yaw because as body rotates CW, feet appear to orbit CCW in body frame
        theta = -self.accumulated_yaw + self.angular_offsets[leg_name]
        
        # Circular orbit at extended radius
        radius = self.foot_radii[leg_name]
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        # CRITICAL FIX: Compensate for base descent in body-frame z-coordinate
        # As base descends in world frame, body-frame foot z must INCREASE to maintain ground contact
        # accumulated_descent is negative (descending), so subtracting it raises the body-frame z
        z = self.foot_z_offsets[leg_name] - self.accumulated_descent
        
        return np.array([x, y, z])