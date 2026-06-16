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
        
        # Base foot positions in BODY frame - lifted significantly to prevent ground penetration
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            pos = v.copy()
            pos[2] += 0.08  # Increased lift to accommodate radial/tangential motion and vertical oscillation
            self.base_feet_pos_body[k] = pos
        
        # Spiral motion parameters - reduced to stay within kinematic workspace
        self.radial_retraction_amplitude = 0.055  # Reduced from 0.08 to avoid joint limits
        self.rotation_sweep_amplitude = 0.09  # Reduced from 0.12 to avoid joint limits
        
        # Vertical motion parameters
        self.vertical_compression_amplitude = 0.09  # Increased to match integrated base displacement
        
        # Base motion parameters
        self.forward_velocity = 0.3  # Constant forward velocity (m/s)
        self.vertical_velocity_amplitude = 0.08  # Vertical oscillation velocity amplitude (m/s)
        self.pitch_rate_amplitude = 0.02  # Reduced significantly to minimize rear leg strain
        
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
        - [0.0, 0.25]: descend, minimal pitch
        - [0.25, 0.5]: ascend, minimal pitch
        - [0.5, 0.75]: maintain height, neutral pitch
        - [0.75, 1.0]: descend, minimal pitch
        """
        
        # Constant forward velocity throughout cycle
        vx = self.forward_velocity
        vy = 0.0
        
        # Smooth continuous vertical velocity using sinusoidal function over full cycle
        phase_shifted = (phase + 0.25) % 1.0
        vz_base = -self.vertical_velocity_amplitude * np.cos(2.0 * np.pi * phase_shifted)
        
        # Apply envelope to create hold phase at 0.5-0.75 with smooth transitions
        if 0.5 <= phase < 0.75:
            hold_factor = 0.0
        elif 0.45 <= phase < 0.5:
            local_t = (phase - 0.45) / 0.05
            hold_factor = 1.0 - smooth_step(local_t)
        elif 0.75 <= phase < 0.8:
            local_t = (phase - 0.75) / 0.05
            hold_factor = smooth_step(local_t)
        else:
            hold_factor = 1.0
        
        vz = vz_base * hold_factor
        
        # Minimal pitch rate synchronized with vertical motion to reduce rear leg strain
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
            # Outward extension: retraction decreases smoothly
            local_t = leg_phase / 0.25
            retraction_factor = 1.0 - smooth_step(local_t)
        elif leg_phase < 0.5:
            # Inward compression: retraction increases smoothly
            local_t = (leg_phase - 0.25) / 0.25
            retraction_factor = smooth_step(local_t)
        elif leg_phase < 0.75:
            # Maximum retraction hold
            retraction_factor = 1.0
        else:
            # Outward recovery: retraction decreases smoothly
            local_t = (leg_phase - 0.75) / 0.25
            retraction_factor = 1.0 - smooth_step(local_t)
        
        # Compute rotation angle for spiral sweep
        rotation_angle = 2.0 * np.pi * leg_phase
        
        # Radial component: move foot toward/away from body centerline
        radial_direction_xy = np.array([base_pos[0], base_pos[1]])
        radial_distance = np.linalg.norm(radial_direction_xy)
        
        if radial_distance > 1e-6:
            radial_unit = radial_direction_xy / radial_distance
        else:
            radial_unit = np.array([1.0, 0.0])
        
        # Apply radial retraction with smooth interpolation
        radial_offset = -retraction_factor * self.radial_retraction_amplitude
        foot[0] += radial_offset * radial_unit[0]
        foot[1] += radial_offset * radial_unit[1]
        
        # Tangential component: rotate foot position around mounting point
        tangent_unit = np.array([-radial_unit[1], radial_unit[0]])
        
        # Apply rotational sweep with sinusoidal modulation for smooth spiral
        sweep_amplitude = self.rotation_sweep_amplitude * np.sin(rotation_angle)
        foot[0] += sweep_amplitude * tangent_unit[0]
        foot[1] += sweep_amplitude * tangent_unit[1]
        
        # Vertical component: approximate integrated base height for proper ground contact
        # Base rises during phase 0.25-0.5, stays high during 0.5-0.75, descends during 0.75-1.0 and 0.0-0.25
        # Feet must move DOWN in body frame (negative offset) when base is HIGH to maintain ground contact
        
        # Approximate base height using smooth phase-based function that mirrors velocity integration
        if leg_phase < 0.25:
            # Base descending from neutral to minimum
            local_t = leg_phase / 0.25
            base_height_factor = -smooth_step(local_t) * 0.5  # Goes from 0 to -0.5
        elif leg_phase < 0.5:
            # Base ascending from minimum to maximum
            local_t = (leg_phase - 0.25) / 0.25
            base_height_factor = -0.5 + smooth_step(local_t) * 1.5  # Goes from -0.5 to +1.0
        elif leg_phase < 0.75:
            # Base at maximum height (hold phase)
            base_height_factor = 1.0
        else:
            # Base descending from maximum toward neutral
            local_t = (leg_phase - 0.75) / 0.25
            base_height_factor = 1.0 - smooth_step(local_t) * 1.0  # Goes from 1.0 to 0
        
        # Feet must compensate inversely: negative offset when base is high, positive when base is low
        vertical_offset = -base_height_factor * self.vertical_compression_amplitude
        foot[2] += vertical_offset
        
        return foot


def smooth_step(t):
    """Smooth interpolation function for C1 continuity"""
    t_clamped = np.clip(t, 0.0, 1.0)
    return t_clamped * t_clamped * (3.0 - 2.0 * t_clamped)