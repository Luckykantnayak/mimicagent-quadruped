from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_SPIRAL_RETRACT_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Spiral retract and advance gait.
    
    All four legs simultaneously spiral inward during stance compression and 
    outward during recovery extension, creating a spring-like compression cycle 
    with helical leg motion while the base translates smoothly forward.
    
    Phase breakdown:
    - [0.0, 0.25]: Outward extension descent - legs extend radially, base descends
    - [0.25, 0.5]: Inward spiral compression - legs spiral inward helically, base rises
    - [0.5, 0.75]: Retracted sustain - legs held retracted, base at peak height
    - [0.75, 1.0]: Outward spiral extension - legs spiral outward, base descends
    
    All legs maintain continuous ground contact throughout the entire cycle.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions (extended stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Store initial ground-level Z in world frame (constant for all feet)
        # Assume all feet start at same ground level
        self.ground_z_world = self.base_feet_pos_body[leg_names[0]][2]
        
        # Motion parameters
        self.radial_retraction_amount = 0.12  # Reduced to avoid joint limits
        self.spiral_tangential_amplitude = 0.06  # Reduced for smoother motion
        
        # Base motion parameters
        self.vx_forward = 0.4  # Slightly reduced for stability
        self.base_height_amplitude = 0.06  # Vertical oscillation amplitude (meters)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track initial base height for compensation
        self.initial_base_height = 0.0
        
        # Spiral handedness for each leg (to maintain symmetry)
        # FL and RR rotate clockwise inward, FR and RL rotate counter-clockwise inward
        self.spiral_direction = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.spiral_direction[leg] = 1.0  # Clockwise when spiraling inward
            else:  # FR or RL
                self.spiral_direction[leg] = -1.0  # Counter-clockwise when spiraling inward

    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent vertical position and constant forward velocity.
        
        Phase-dependent vertical motion:
        - [0.0, 0.25]: Descend (base lowers)
        - [0.25, 0.5]: Ascend (base rises)
        - [0.5, 0.75]: Hold at peak (constant height)
        - [0.75, 1.0]: Descend (base lowers)
        """
        
        # Track initial base height on first call
        if self.t == 0.0:
            self.initial_base_height = self.root_pos[2]
        
        # Compute target base height offset based on phase
        if phase < 0.25:
            # Descend during outward extension
            sub_phase = phase / 0.25
            # Smooth descent: 0 -> -amplitude
            height_offset = -self.base_height_amplitude * 0.5 * (1.0 - np.cos(np.pi * sub_phase))
        elif phase < 0.5:
            # Ascend during inward compression
            sub_phase = (phase - 0.25) / 0.25
            # Smooth ascent: -amplitude -> +amplitude
            height_offset = -self.base_height_amplitude + 2.0 * self.base_height_amplitude * 0.5 * (1.0 - np.cos(np.pi * sub_phase))
        elif phase < 0.75:
            # Hold at peak during retraction
            height_offset = self.base_height_amplitude
        else:
            # Descend during outward spiral extension
            sub_phase = (phase - 0.75) / 0.25
            # Smooth descent: +amplitude -> 0
            height_offset = self.base_height_amplitude * 0.5 * (1.0 + np.cos(np.pi * sub_phase))
        
        # Compute velocity from height derivative
        # Use finite difference approximation with small epsilon
        epsilon = 0.001
        phase_next = (phase + epsilon) % 1.0
        
        if phase_next < 0.25:
            sub_phase_next = phase_next / 0.25
            height_offset_next = -self.base_height_amplitude * 0.5 * (1.0 - np.cos(np.pi * sub_phase_next))
        elif phase_next < 0.5:
            sub_phase_next = (phase_next - 0.25) / 0.25
            height_offset_next = -self.base_height_amplitude + 2.0 * self.base_height_amplitude * 0.5 * (1.0 - np.cos(np.pi * sub_phase_next))
        elif phase_next < 0.75:
            height_offset_next = self.base_height_amplitude
        else:
            sub_phase_next = (phase_next - 0.75) / 0.25
            height_offset_next = self.base_height_amplitude * 0.5 * (1.0 + np.cos(np.pi * sub_phase_next))
        
        vz = (height_offset_next - height_offset) / epsilon * self.freq
        
        # Set velocities in world frame
        self.vel_world = np.array([self.vx_forward, 0.0, vz])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
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
        Compute foot position in body frame with helical spiral motion.
        
        All legs move synchronously:
        - Radial component: extension/retraction relative to base position
        - Tangential component: spiral motion perpendicular to radial direction
        - Vertical component: compensates for base height changes to maintain ground contact
        
        Maintains continuous ground contact throughout.
        """
        
        # Start from base extended stance position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial direction from body center to foot (in XY plane)
        radial_xy = np.array([foot[0], foot[1]])
        radial_distance = np.linalg.norm(radial_xy)
        
        if radial_distance < 1e-6:
            # Foot at body center, use default direction based on leg name
            if 'F' in leg_name and 'L' in leg_name:
                radial_unit = np.array([1.0, 1.0]) / np.sqrt(2.0)
            elif 'F' in leg_name and 'R' in leg_name:
                radial_unit = np.array([1.0, -1.0]) / np.sqrt(2.0)
            elif 'R' in leg_name and 'L' in leg_name:
                radial_unit = np.array([-1.0, 1.0]) / np.sqrt(2.0)
            else:
                radial_unit = np.array([-1.0, -1.0]) / np.sqrt(2.0)
        else:
            radial_unit = radial_xy / radial_distance
        
        # Tangent direction (perpendicular to radial in XY plane)
        tangent_unit = np.array([-radial_unit[1], radial_unit[0]])
        
        # Phase-dependent radial and tangential displacements
        radial_offset = 0.0
        tangent_offset = 0.0
        
        if phase < 0.25:
            # [0.0, 0.25]: Outward extension descent
            # Smoothly extend outward from retracted position
            sub_phase = phase / 0.25
            blend = 0.5 * (1.0 - np.cos(np.pi * sub_phase))  # Smooth 0 -> 1
            radial_offset = -self.radial_retraction_amount * (1.0 - blend)
            tangent_offset = 0.0  # No spiral during pure extension
            
        elif phase < 0.5:
            # [0.25, 0.5]: Inward spiral compression
            # Spiral inward with helical motion
            sub_phase = (phase - 0.25) / 0.25
            blend = 0.5 * (1.0 - np.cos(np.pi * sub_phase))  # Smooth 0 -> 1
            radial_offset = -self.radial_retraction_amount * blend
            # Spiral component: tangential motion during compression
            spiral_progress = np.sin(np.pi * sub_phase)
            tangent_offset = self.spiral_direction[leg_name] * self.spiral_tangential_amplitude * spiral_progress
            
        elif phase < 0.75:
            # [0.5, 0.75]: Retracted sustain
            # Hold at maximum retraction
            radial_offset = -self.radial_retraction_amount
            tangent_offset = 0.0
            
        else:
            # [0.75, 1.0]: Outward spiral extension
            # Spiral outward (reverse helical motion)
            sub_phase = (phase - 0.75) / 0.25
            blend = 0.5 * (1.0 - np.cos(np.pi * sub_phase))  # Smooth 0 -> 1
            radial_offset = -self.radial_retraction_amount * (1.0 - blend)
            # Reverse spiral component: tangential motion during extension
            spiral_progress = np.sin(np.pi * sub_phase)
            tangent_offset = -self.spiral_direction[leg_name] * self.spiral_tangential_amplitude * spiral_progress
        
        # Apply radial offset
        foot[0] += radial_offset * radial_unit[0]
        foot[1] += radial_offset * radial_unit[1]
        
        # Apply tangential (spiral) offset
        foot[0] += tangent_offset * tangent_unit[0]
        foot[1] += tangent_offset * tangent_unit[1]
        
        # Compensate Z coordinate for base height changes to maintain ground contact
        # Compute current base height change from initial
        base_height_change = self.root_pos[2] - self.initial_base_height
        
        # In body frame, if base moves up by delta_z, foot must move down by delta_z
        # to maintain constant world Z position at ground level
        foot[2] = self.ground_z_world - base_height_change
        
        return foot