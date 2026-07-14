from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_SPIRAL_RETRACT_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Forward locomotion with all four legs simultaneously spiraling inward during 
    compression and extending outward during recovery, creating spring-like 
    corkscrew motion while the base translates smoothly forward with vertical oscillation.
    
    Phase structure:
    - [0.0, 0.25]: outward_extension_descent - legs extend radially outward in spiral, base descends
    - [0.25, 0.5]: inward_spiral_compression - legs spiral inward, base rises
    - [0.5, 0.75]: maximum_retraction_hold - legs held at minimum radial distance, base at peak height
    - [0.75, 1.0]: outward_spiral_recovery - legs begin outward spiral extension, base descends
    
    All four legs execute identical synchronized spiral trajectories with zero phase offset.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Motion cycle frequency (Hz)
        
        # Base foot positions in BODY frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Spiral motion parameters
        self.radial_retraction_amplitude = 0.08  # Maximum inward radial displacement (m)
        self.rotation_sweep_amplitude = 0.12  # Tangential sweep amplitude (radians)
        
        # Vertical motion parameters
        self.vertical_compression_amplitude = 0.04  # Vertical foot retraction during compression (m)
        
        # Base motion parameters
        self.forward_velocity = 0.3  # Constant forward velocity (m/s)
        self.vertical_velocity_amplitude = 0.10  # Vertical oscillation velocity amplitude (m/s)
        self.pitch_rate_amplitude = 0.12  # Pitch modulation amplitude (rad/s)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # All legs synchronized with zero phase offset
        self.phase_offsets = {
            leg: 0.0 for leg in self.leg_names
        }

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        Phase-dependent behavior:
        - [0.0, 0.25]: descend, slight nose-down pitch
        - [0.25, 0.5]: ascend, slight nose-up pitch
        - [0.5, 0.75]: maintain height, neutral pitch
        - [0.75, 1.0]: descend, slight nose-down pitch
        """
        
        # Constant forward velocity throughout cycle
        vx = self.forward_velocity
        vy = 0.0
        
        # Vertical velocity based on phase with smooth transitions
        if phase < 0.25:
            # Descending during outward extension
            local_phase = phase / 0.25
            vz = -self.vertical_velocity_amplitude * np.sin(np.pi * local_phase)
            pitch_rate = -self.pitch_rate_amplitude * np.sin(np.pi * local_phase)
        elif phase < 0.5:
            # Ascending during inward compression
            local_phase = (phase - 0.25) / 0.25
            vz = self.vertical_velocity_amplitude * np.sin(np.pi * local_phase)
            pitch_rate = self.pitch_rate_amplitude * np.sin(np.pi * local_phase)
        elif phase < 0.75:
            # Maintain height during maximum retraction
            vz = 0.0
            pitch_rate = 0.0
        else:
            # Descending during outward recovery
            local_phase = (phase - 0.75) / 0.25
            vz = -self.vertical_velocity_amplitude * np.sin(np.pi * local_phase)
            pitch_rate = -self.pitch_rate_amplitude * np.sin(np.pi * local_phase)
        
        # Set velocity commands in WORLD frame
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
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
        Compute spiral foot trajectory in BODY frame with vertical modulation.
        
        Combines three motion components:
        1. Radial distance modulation (retraction/extension from centerline)
        2. Rotational sweeping around nominal leg mounting point
        3. Vertical compression/extension to maintain ground contact during base oscillation
        
        All legs execute identical trajectories with zero phase offset.
        """
        
        # Apply phase offset (zero for all legs in this skill)
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Compute radial retraction factor (0 = extended, 1 = fully retracted)
        if leg_phase < 0.25:
            # Outward extension: retraction decreases
            retraction_factor = 1.0 - (leg_phase / 0.25)
        elif leg_phase < 0.5:
            # Inward compression: retraction increases
            local_phase = (leg_phase - 0.25) / 0.25
            retraction_factor = local_phase
        elif leg_phase < 0.75:
            # Maximum retraction hold
            retraction_factor = 1.0
        else:
            # Outward recovery: retraction decreases
            local_phase = (leg_phase - 0.75) / 0.25
            retraction_factor = 1.0 - local_phase
        
        # Smooth radial retraction using sinusoidal interpolation
        retraction_smooth = 0.5 * (1.0 - np.cos(np.pi * retraction_factor))
        
        # Compute rotation angle for spiral sweep
        # Complete one full rotation cycle over phase [0, 1]
        rotation_angle = 2.0 * np.pi * leg_phase
        
        # Radial component: move foot toward/away from body centerline
        # Compute direction from body center to base foot position
        radial_direction_xy = np.array([base_pos[0], base_pos[1]])
        radial_distance = np.linalg.norm(radial_direction_xy)
        
        if radial_distance > 1e-6:
            radial_unit = radial_direction_xy / radial_distance
        else:
            radial_unit = np.array([1.0, 0.0])
        
        # Apply radial retraction
        radial_offset = -retraction_smooth * self.radial_retraction_amplitude
        foot[0] += radial_offset * radial_unit[0]
        foot[1] += radial_offset * radial_unit[1]
        
        # Tangential component: rotate foot position around mounting point
        # Compute tangent direction (perpendicular to radial)
        tangent_unit = np.array([-radial_unit[1], radial_unit[0]])
        
        # Apply rotational sweep with sinusoidal modulation
        sweep_amplitude = self.rotation_sweep_amplitude * np.sin(rotation_angle)
        foot[0] += sweep_amplitude * tangent_unit[0]
        foot[1] += sweep_amplitude * tangent_unit[1]
        
        # Vertical component: modulate foot height to maintain ground contact
        # Feet compress (rise in body frame) when base rises and legs retract
        # Feet extend (lower in body frame) when base descends and legs extend
        # This creates spring-like compression while maintaining ground contact
        
        # Compute vertical compression factor synchronized with retraction
        # Maximum compression occurs at maximum retraction (phase 0.5-0.75)
        if leg_phase < 0.25:
            # Extension phase: feet lower as legs extend outward and base descends
            local_phase = leg_phase / 0.25
            vertical_factor = 1.0 - local_phase
        elif leg_phase < 0.5:
            # Compression phase: feet rise as legs retract inward and base ascends
            local_phase = (leg_phase - 0.25) / 0.25
            vertical_factor = local_phase
        elif leg_phase < 0.75:
            # Hold phase: feet at maximum compression height
            vertical_factor = 1.0
        else:
            # Recovery phase: feet begin lowering as legs extend and base descends
            local_phase = (leg_phase - 0.75) / 0.25
            vertical_factor = 1.0 - local_phase
        
        # Smooth vertical transition
        vertical_smooth = 0.5 * (1.0 - np.cos(np.pi * vertical_factor))
        
        # Apply vertical offset (positive = foot rises in body frame during compression)
        vertical_offset = vertical_smooth * self.vertical_compression_amplitude
        foot[2] += vertical_offset
        
        return foot