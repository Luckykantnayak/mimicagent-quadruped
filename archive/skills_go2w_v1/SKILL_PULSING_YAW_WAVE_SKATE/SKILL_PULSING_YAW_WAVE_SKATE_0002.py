from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_PULSING_YAW_WAVE_SKATE_MotionGenerator(BaseMotionGenerator):
    """
    Serpentine skating motion with synchronized leg extension/contraction and yaw oscillation.
    
    - All four wheels remain in continuous ground contact (skating, no aerial phase)
    - Legs extend/contract together with front-rear phase offset creating traveling wave
    - Body rises during extension with positive yaw (right carve on outer wheels)
    - Body drops during contraction with negative yaw (left slide with wheel unloading)
    - Forward velocity sustained throughout for skating momentum
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Leg extension/contraction parameters
        self.extension_amplitude = 0.12  # radial extension range (meters) - reduced for joint safety
        self.extension_vertical_lift = 0.03  # upward lift in body frame during extension to maintain ground contact
        
        # Phase offsets for traveling wave effect (front legs lead by 0.125 phase units)
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('F'):  # Front legs lead
                self.phase_offsets[leg] = 0.0
            else:  # Rear legs lag
                self.phase_offsets[leg] = 0.125
        
        # Base motion parameters
        self.vx_forward = 1.5  # constant forward skating velocity (m/s)
        self.vz_amplitude = 0.3  # vertical velocity oscillation amplitude (m/s)
        self.yaw_rate_amplitude = 2.5  # yaw rate oscillation amplitude (rad/s)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity, oscillating vertical velocity, and oscillating yaw rate.
        
        Phase structure:
        - [0.0, 0.25]: extension, body rising (vz > 0), yaw positive (right carve)
        - [0.25, 0.375]: extended transition, vz → 0, yaw rate → 0
        - [0.375, 0.625]: contraction, body dropping (vz < 0), yaw negative (left slide)
        - [0.625, 0.75]: contracted transition, vz → 0, yaw rate → 0
        - [0.75, 1.0]: pre-extension wave, vz > 0 again, yaw positive again
        """
        
        # Constant forward velocity for skating momentum
        vx = self.vx_forward
        
        # Vertical velocity: smooth oscillation with peaks at extension, troughs at contraction
        # Use sinusoidal profile with phase shift so vz peaks around phase 0.125 and troughs at 0.5
        vz_phase = (phase - 0.125) * 2 * np.pi
        vz = self.vz_amplitude * np.sin(vz_phase)
        
        # Yaw rate: oscillates positive (right carve) during extension, negative (left slide) during contraction
        # Use sinusoidal profile: positive peak around phase 0.125, negative peak around 0.5
        yaw_phase = (phase - 0.125) * 2 * np.pi
        yaw_rate = self.yaw_rate_amplitude * np.sin(yaw_phase)
        
        # Set velocity commands
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
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
        Compute foot position in body frame with radial extension/contraction.
        
        Front legs lead extension/contraction by phase offset to create traveling wave.
        Feet remain at ground level (z_world = 0) by adjusting z in body frame as base rises/falls.
        
        Extension pattern:
        - Phase [0.0, 0.5]: extending (legs lengthen radially outward, z in body frame increases)
        - Phase [0.5, 1.0]: contracting (legs shorten radially inward, z in body frame decreases)
        """
        
        # Apply phase offset for traveling wave
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Get base foot position
        foot_base = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial direction from body origin to foot (in x-y plane, ignore z)
        radial_xy = np.array([foot_base[0], foot_base[1]])
        radial_dist_xy = np.linalg.norm(radial_xy)
        if radial_dist_xy > 1e-6:
            radial_dir_xy = radial_xy / radial_dist_xy
        else:
            radial_dir_xy = np.array([1.0, 0.0])
        
        # Extension/contraction cycle: smooth sinusoidal modulation
        # leg_phase 0.0 → mid extension
        # leg_phase 0.25 → max extension
        # leg_phase 0.5 → mid contraction
        # leg_phase 0.75 → min extension
        extension_cycle_phase = leg_phase * 2 * np.pi
        extension_factor = np.sin(extension_cycle_phase)  # ranges [-1, 1]
        
        # Radial offset: positive = extend outward, negative = contract inward
        radial_offset_xy = self.extension_amplitude * extension_factor * radial_dir_xy
        
        # Vertical offset: feet move upward in body frame during extension to compensate for base rise
        # This maintains ground contact (z_world ≈ 0) while creating visible leg length variation
        vertical_offset = self.extension_vertical_lift * extension_factor
        
        # Compute final foot position
        foot = foot_base.copy()
        foot[0] += radial_offset_xy[0]
        foot[1] += radial_offset_xy[1]
        foot[2] += vertical_offset  # positive during extension (upward in body frame)
        
        return foot