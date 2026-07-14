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
        
        # Orbital motion parameters
        self.orbit_radius = 0.06  # meters, radius of circular foot trajectory
        self.forward_velocity = 0.3  # m/s, constant forward base velocity
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Identify leg groups (diagonal pairs)
        self.group_1 = [leg for leg in leg_names if leg.startswith('FL') or leg.startswith('RR')]
        self.group_2 = [leg for leg in leg_names if leg.startswith('FR') or leg.startswith('RL')]
        
        # Compute diagonal midpoints for orbital centers
        self.diagonal_centers = self._compute_diagonal_centers()
        
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
        - Non-orbiting legs remain at base position with minimal adjustment
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine which group this leg belongs to
        if leg_name in self.group_1:
            group_id = 'group_1'
            orbit_phase_range = (0.0, 0.5)
        else:
            group_id = 'group_2'
            orbit_phase_range = (0.5, 1.0)
        
        # Check if this leg is currently orbiting
        if orbit_phase_range[0] <= phase < orbit_phase_range[1]:
            # Leg is orbiting - compute circular trajectory
            foot_pos = self._compute_orbital_position(
                leg_name, 
                phase, 
                orbit_phase_range, 
                group_id
            )
        else:
            # Leg is in stable stance - hold base position
            foot_pos = base_pos
        
        return foot_pos

    def _compute_orbital_position(self, leg_name, phase, orbit_phase_range, group_id):
        """
        Compute circular orbital trajectory around diagonal pair center.
        
        Clockwise motion when viewed from above:
        - Front leg: outward laterally → rearward → inward
        - Rear leg: outward laterally → forward → inward
        """
        # Get base position and diagonal center
        base_pos = self.base_feet_pos_body[leg_name].copy()
        center = self.diagonal_centers[group_id]
        
        # Normalize phase to [0, 1] within the orbital sub-phase
        phase_start, phase_end = orbit_phase_range
        local_phase = (phase - phase_start) / (phase_end - phase_start)
        
        # Compute relative position from center to base position
        rel_pos = base_pos[:2] - center[:2]  # XY plane only
        base_radius = np.linalg.norm(rel_pos)
        base_angle = np.arctan2(rel_pos[1], rel_pos[0])
        
        # Circular motion: full rotation over the sub-phase
        # Clockwise: negative angle increment
        angle = base_angle - 2 * np.pi * local_phase
        
        # Compute new position on circular orbit
        orbit_x = center[0] + self.orbit_radius * np.cos(angle)
        orbit_y = center[1] + self.orbit_radius * np.sin(angle)
        
        # Maintain ground contact (Z = base position Z)
        foot_pos = np.array([orbit_x, orbit_y, base_pos[2]])
        
        return foot_pos