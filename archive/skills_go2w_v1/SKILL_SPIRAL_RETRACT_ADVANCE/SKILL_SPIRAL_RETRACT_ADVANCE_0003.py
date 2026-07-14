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
        
        # Base foot positions in BODY frame - adjusted upward to prevent initial ground penetration
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            pos = v.copy()
            pos[2] += 0.025  # Lift initial foot positions to ensure valid starting configuration
            self.base_feet_pos_body[k] = pos
        
        # Spiral motion parameters
        self.radial_retraction_amplitude = 0.08  # Maximum inward radial displacement (m)
        self.rotation_sweep_amplitude = 0.12  # Tangential sweep amplitude (radians)
        
        # Vertical motion parameters
        self.vertical_compression_amplitude = 0.06  # Vertical foot compensation amplitude (m)
        
        # Base motion parameters
        self.forward_velocity = 0.3  # Constant forward velocity (m/s)
        self.vertical_velocity_amplitude = 0.08  # Vertical oscillation velocity amplitude (m/s)
        self.pitch_rate_amplitude = 0.06  # Pitch modulation amplitude (rad/s) - reduced to avoid rear leg issues
        
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
        
        # Smooth continuous vertical velocity using sinusoidal function over full cycle
        # Descends in phases 0-0.25 and 0.75-1.0, ascends in 0.25-0.5, holds in 0.5-0.75
        phase_shifted = (phase + 0.25) % 1.0  # Shift to align descent with phase 0
        vz_base = -self.vertical_velocity_amplitude * np.cos(2.0 * np.pi * phase_shifted)
        
        # Apply envelope to create hold phase at 0.5-0.75
        if 0.5 <= phase < 0.75:
            hold_factor = 0.0
        elif 0.45 <= phase < 0.5:
            # Smooth transition into hold
            local_t = (phase - 0.45) / 0.05
            hold_factor = 1.0 - smooth_step(local_t)
        elif 0.75 <= phase < 0.8:
            # Smooth transition out of hold
            local_t = (phase - 0.75) / 0.05
            hold_factor = smooth_step(local_t)
        else:
            hold_factor = 1.0
        
        vz = vz_base * hold_factor
        
        # Pitch rate synchronized with vertical motion - reduced amplitude
        if phase < 0.25:
            local_phase = phase / 0.25
            pitch_rate = -self.pitch_rate_amplitude * np.sin(np.pi * local_phase)
        elif phase < 0.5:
            local_phase = (phase - 0.25) / 0.25
            pitch_rate = self.pitch_rate_amplitude * np.sin(np.pi * local_phase)
        elif phase < 0.75:
            pitch_rate = 0.0
        else:
            local_phase = (phase - 0.75) / 0.25
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
        3. Vertical compensation to maintain ground contact during base oscillation
        
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
        
        # Compute rotation angle for spiral sweep with smoother progression
        rotation_angle = 2.0 * np.pi * leg_phase
        
        # Radial component: move foot toward/away from body centerline
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
        tangent_unit = np.array([-radial_unit[1], radial_unit[0]])
        
        # Apply rotational sweep with sinusoidal modulation
        sweep_amplitude = self.rotation_sweep_amplitude * np.sin(rotation_angle)
        foot[0] += sweep_amplitude * tangent_unit[0]
        foot[1] += sweep_amplitude * tangent_unit[1]
        
        # Vertical component: INVERTED to maintain ground contact
        # When base rises (phases 0.25-0.75), feet must move DOWN in body frame
        # When base descends (phases 0.0-0.25, 0.75-1.0), feet must move UP in body frame
        # This compensates for base vertical motion to keep feet on ground in world frame
        
        if leg_phase < 0.25:
            # Base descending: feet rise slightly in body frame (positive offset)
            local_phase = leg_phase / 0.25
            vertical_factor = -local_phase  # Starts at 0, goes to -1
        elif leg_phase < 0.5:
            # Base ascending: feet lower in body frame (negative offset)
            local_phase = (leg_phase - 0.25) / 0.25
            vertical_factor = -1.0 + local_phase  # Goes from -1 to 0
        elif leg_phase < 0.75:
            # Base at peak: feet at neutral position
            vertical_factor = 0.0
        else:
            # Base descending: feet rise in body frame
            local_phase = (leg_phase - 0.75) / 0.25
            vertical_factor = -local_phase  # Goes from 0 to -1
        
        # Smooth vertical transition with phase continuity
        vertical_smooth = smooth_step(abs(vertical_factor)) * np.sign(vertical_factor) if vertical_factor != 0 else 0.0
        
        # Apply inverted vertical offset (negative = foot lowers in body frame when base rises)
        # This maintains ground contact while creating visual compression effect
        vertical_offset = vertical_smooth * self.vertical_compression_amplitude
        foot[2] += vertical_offset
        
        return foot


def smooth_step(t):
    """Smooth interpolation function for C1 continuity"""
    return t * t * (3.0 - 2.0 * t)