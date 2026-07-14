from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ORBITING_SATELLITES_MotionGenerator(BaseMotionGenerator):
    """
    Orbiting Satellites Gait:
    
    Forward locomotion via alternating diagonal pairs that orbit circularly 
    around their shared diagonal midpoint while maintaining ground contact.
    
    - Phase 0.0-0.5: FL+RR orbit clockwise around their midpoint; FR+RL stable
    - Phase 0.5-1.0: FR+RL orbit clockwise around their midpoint; FL+RR stable
    - All feet remain grounded throughout
    - Base moves forward with constant velocity
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Hz, gait cycle frequency
        
        # Orbital motion parameters - constrained for kinematic safety
        self.forward_velocity = 0.3  # m/s, constant forward base velocity
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Identify leg groups (diagonal pairs)
        self.group_1 = [leg for leg in leg_names if leg.startswith('FL') or leg.startswith('RR')]
        self.group_2 = [leg for leg in leg_names if leg.startswith('FR') or leg.startswith('RL')]
        
        # Compute diagonal midpoints for orbital centers
        self.diagonal_centers = self._compute_diagonal_centers()
        
        # Compute individual base radius for each leg to preserve natural geometry
        self.base_radii = self._compute_base_radii()
        self.base_angles = self._compute_base_angles()
        
        # Per-leg orbital parameters constrained by kinematic workspace
        self.orbit_params = self._compute_safe_orbit_params()
        
        # Phase transition blending window (fraction of phase)
        self.transition_window = 0.08
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def _compute_diagonal_centers(self):
        """
        Compute the geometric center of each diagonal pair in body frame.
        """
        centers = {}
        
        # Group 1: FL and RR
        fl_pos = self.base_feet_pos_body[self.group_1[0]]
        rr_pos = self.base_feet_pos_body[self.group_1[1]]
        centers['group_1'] = (fl_pos + rr_pos) / 2.0
        
        # Group 2: FR and RL
        fr_pos = self.base_feet_pos_body[self.group_2[0]]
        rl_pos = self.base_feet_pos_body[self.group_2[1]]
        centers['group_2'] = (fr_pos + rl_pos) / 2.0
        
        return centers

    def _compute_base_radii(self):
        """
        Compute the natural distance from each foot to its diagonal center.
        This preserves individual leg geometry during orbital motion.
        """
        radii = {}
        
        for leg_name in self.leg_names:
            if leg_name in self.group_1:
                group_id = 'group_1'
            else:
                group_id = 'group_2'
            
            base_pos = self.base_feet_pos_body[leg_name][:2]
            center = self.diagonal_centers[group_id][:2]
            rel_pos = base_pos - center
            radii[leg_name] = np.linalg.norm(rel_pos)
        
        return radii

    def _compute_base_angles(self):
        """
        Compute the initial angle of each foot relative to its diagonal center.
        This ensures orbits start from the natural stance position.
        """
        angles = {}
        
        for leg_name in self.leg_names:
            if leg_name in self.group_1:
                group_id = 'group_1'
            else:
                group_id = 'group_2'
            
            base_pos = self.base_feet_pos_body[leg_name][:2]
            center = self.diagonal_centers[group_id][:2]
            rel_pos = base_pos - center
            angles[leg_name] = np.arctan2(rel_pos[1], rel_pos[0])
        
        return angles

    def _compute_safe_orbit_params(self):
        """
        Compute safe orbital parameters for each leg based on position and type.
        Front legs can orbit with larger radius and angular range than rear legs.
        """
        params = {}
        
        for leg_name in self.leg_names:
            # Differentiate front vs rear legs
            is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
            
            if is_front:
                # Front legs: larger orbital radius and angular range
                orbit_radius = 0.012  # meters
                angular_range = np.pi / 3.0  # ±30 degrees
                z_compliance = 0.008  # Allow 8mm vertical variation
            else:
                # Rear legs: smaller orbital radius and angular range
                orbit_radius = 0.008  # meters
                angular_range = np.pi / 4.5  # ±20 degrees
                z_compliance = 0.006  # Allow 6mm vertical variation
            
            params[leg_name] = {
                'orbit_radius': orbit_radius,
                'angular_range': angular_range,
                'z_compliance': z_compliance
            }
        
        return params

    def update_base_motion(self, phase, dt):
        """
        Constant forward velocity throughout the gait cycle.
        No angular velocity (no roll, pitch, or yaw rotation).
        """
        # Constant forward velocity in world frame
        self.vel_world = np.array([self.forward_velocity, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        # Integrate base pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on current phase.
        
        - Group 1 (FL, RR) orbits during phase 0.0-0.5
        - Group 2 (FR, RL) orbits during phase 0.5-1.0
        - Smooth transitions at phase boundaries using blending
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine which group this leg belongs to
        if leg_name in self.group_1:
            group_id = 'group_1'
            orbit_phase_range = (0.0, 0.5)
        else:
            group_id = 'group_2'
            orbit_phase_range = (0.5, 1.0)
        
        # Compute transition blending factor
        blend_factor = self._compute_transition_blend(phase, orbit_phase_range)
        
        if blend_factor > 0.0:
            # Leg is orbiting or transitioning - compute orbital position
            orbital_pos = self._compute_orbital_position(
                leg_name, 
                phase, 
                orbit_phase_range, 
                group_id
            )
            
            # Blend between base and orbital position during transitions
            if blend_factor < 1.0:
                foot_pos = base_pos * (1.0 - blend_factor) + orbital_pos * blend_factor
            else:
                foot_pos = orbital_pos
        else:
            # Leg is in stable stance - hold base position
            foot_pos = base_pos
        
        return foot_pos

    def _compute_transition_blend(self, phase, orbit_phase_range):
        """
        Compute smooth blending factor for phase transitions.
        Returns 0.0 outside active range, 1.0 during stable orbit, 
        and smoothly interpolates during entry/exit windows.
        """
        phase_start, phase_end = orbit_phase_range
        
        # Normalize to phase range [0, 1]
        if phase < phase_start or phase >= phase_end:
            return 0.0
        
        local_phase = (phase - phase_start) / (phase_end - phase_start)
        
        # Entry transition
        if local_phase < self.transition_window:
            blend = local_phase / self.transition_window
            return 0.5 * (1.0 - np.cos(np.pi * blend))  # Smooth cosine ramp up
        
        # Exit transition
        elif local_phase > (1.0 - self.transition_window):
            exit_progress = (local_phase - (1.0 - self.transition_window)) / self.transition_window
            blend = 1.0 - exit_progress
            return 0.5 * (1.0 - np.cos(np.pi * blend))  # Smooth cosine ramp down
        
        # Stable orbital phase
        else:
            return 1.0

    def _compute_orbital_position(self, leg_name, phase, orbit_phase_range, group_id):
        """
        Compute constrained circular orbital trajectory around diagonal pair center.
        
        Uses partial angular arc with leg-specific radius to stay within joint limits.
        Allows small Z-axis variation to respect kinematic coupling.
        """
        # Get base position and diagonal center
        base_pos = self.base_feet_pos_body[leg_name].copy()
        center = self.diagonal_centers[group_id]
        
        # Get safe orbit parameters for this leg
        orbit_params = self.orbit_params[leg_name]
        orbit_radius = orbit_params['orbit_radius']
        angular_range = orbit_params['angular_range']
        z_compliance = orbit_params['z_compliance']
        
        # Normalize phase to [0, 1] within the orbital sub-phase
        phase_start, phase_end = orbit_phase_range
        local_phase = (phase - phase_start) / (phase_end - phase_start)
        
        # Get natural radius and angle for this leg
        base_radius = self.base_radii[leg_name]
        base_angle = self.base_angles[leg_name]
        
        # Constrained angular excursion (partial arc, not full 360 degrees)
        # Oscillate within ±angular_range around base angle
        angle_offset = angular_range * np.sin(2 * np.pi * local_phase)
        angle = base_angle - angle_offset  # Clockwise motion
        
        # Constant orbital radius (no modulation)
        effective_radius = base_radius + orbit_radius * np.sin(np.pi * local_phase)
        
        # Compute new position on constrained circular arc
        orbit_x = center[0] + effective_radius * np.cos(angle)
        orbit_y = center[1] + effective_radius * np.sin(angle)
        
        # Allow small Z-axis variation coupled to radial extension
        # When radius extends, foot naturally lifts slightly
        z_offset = z_compliance * np.sin(np.pi * local_phase)
        orbit_z = base_pos[2] + z_offset
        
        foot_pos = np.array([orbit_x, orbit_y, orbit_z])
        
        return foot_pos