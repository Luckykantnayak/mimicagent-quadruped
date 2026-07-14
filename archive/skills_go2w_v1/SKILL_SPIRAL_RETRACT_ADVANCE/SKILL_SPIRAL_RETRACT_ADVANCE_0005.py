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
        
        # Motion parameters
        self.radial_retraction_amount = 0.15  # How much to pull feet inward (meters)
        self.spiral_tangential_amplitude = 0.08  # Tangential spiral component (meters)
        
        # Base motion parameters
        self.vx_forward = 0.5  # Constant forward velocity
        self.vz_amplitude = 0.3  # Vertical oscillation amplitude
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
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
        Update base pose using phase-dependent vertical velocity and constant forward velocity.
        
        Phase-dependent vertical motion:
        - [0.0, 0.25]: Descend (negative vz)
        - [0.25, 0.5]: Ascend (positive vz)
        - [0.5, 0.75]: Hold (zero vz)
        - [0.75, 1.0]: Descend (negative vz)
        """
        
        # Constant forward velocity
        vx = self.vx_forward
        
        # Phase-dependent vertical velocity
        if phase < 0.25:
            # Outward extension descent: descend smoothly
            sub_phase = phase / 0.25
            vz = -self.vz_amplitude * np.sin(np.pi * sub_phase)
        elif phase < 0.5:
            # Inward spiral compression: ascend
            sub_phase = (phase - 0.25) / 0.25
            vz = self.vz_amplitude * np.sin(np.pi * sub_phase)
        elif phase < 0.75:
            # Retracted sustain: hold height
            vz = 0.0
        else:
            # Outward spiral extension: descend
            sub_phase = (phase - 0.75) / 0.25
            vz = -self.vz_amplitude * np.sin(np.pi * sub_phase)
        
        # Set velocities in world frame
        self.vel_world = np.array([vx, 0.0, vz])
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
        
        Maintains continuous ground contact throughout.
        """
        
        # Start from base extended stance position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial direction from body center to foot (in XY plane)
        radial_xy = np.array([foot[0], foot[1]])
        radial_distance = np.linalg.norm(radial_xy)
        
        if radial_distance < 1e-6:
            # Foot at body center, use default direction
            radial_unit = np.array([1.0, 0.0])
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
        
        # Z coordinate remains at ground level (continuous contact)
        # Maintain original z from base position
        
        return foot