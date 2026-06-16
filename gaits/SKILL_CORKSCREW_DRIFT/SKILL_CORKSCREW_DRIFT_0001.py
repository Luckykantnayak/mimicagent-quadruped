from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_CORKSCREW_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Corkscrew drift gait: robot spirals forward in helical trajectory.
    
    - Base executes continuous 360° yaw rotation per cycle
    - Lateral velocity oscillates (right -> left -> neutral)
    - Forward velocity remains constant
    - All four feet maintain sliding ground contact
    - Diagonal pairs alternate extension/tucking in drift phases
    - Synchronized circular paddling during peak yaw phase
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Hz, slower for controlled spiral
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters
        self.vx_constant = 0.5  # m/s forward velocity (constant)
        self.vy_amplitude = 0.3  # m/s lateral drift amplitude
        self.yaw_rate_base = 2.0 * np.pi  # rad/s base yaw rate (360° per cycle)
        self.yaw_rate_peak_multiplier = 1.5  # peak multiplier in paddling phase
        
        # Leg motion parameters
        self.lateral_extension = 0.08  # m, max lateral foot displacement
        self.paddling_radius = 0.06  # m, circular paddling radius
        self.forward_bias = 0.04  # m, forward component during extension
        
        # Phase offsets for diagonal pairs (used in paddling phase)
        self.paddling_phase_offsets = {
            leg_names[0]: 0.0,   # FL
            leg_names[1]: 0.5,   # FR (90° offset in practice)
            leg_names[2]: 0.5,   # RL (90° offset in practice)
            leg_names[3]: 0.0,   # RR
        }

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity, oscillating lateral velocity,
        and phase-dependent yaw rate (peaks during synchronized paddling).
        """
        # Constant forward velocity
        vx = self.vx_constant
        
        # Oscillating lateral velocity: right [0, 0.25], left [0.25, 0.5], diminishing [0.5, 1.0]
        # Use sine wave that goes: 0 -> +peak -> 0 -> -peak -> 0
        vy = self.vy_amplitude * np.sin(2.0 * np.pi * phase)
        
        # No vertical velocity
        vz = 0.0
        
        # Yaw rate: constant baseline, peaks in phase [0.5, 0.75]
        if 0.5 <= phase < 0.75:
            # Peak yaw rate during synchronized paddling
            yaw_rate = self.yaw_rate_base * self.yaw_rate_peak_multiplier
        else:
            # Baseline yaw rate
            yaw_rate = self.yaw_rate_base
        
        # Set world frame velocities
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in BODY frame based on phase and leg assignment.
        
        Phase breakdown:
        [0.0, 0.25]: Rightward drift - FL/RR extend, FR/RL tuck
        [0.25, 0.5]: Leftward drift - FR/RL extend, FL/RR tuck
        [0.5, 0.75]: Synchronized circular paddling (peak yaw)
        [0.75, 1.0]: Stabilization return to neutral
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg group for diagonal pair coordination
        is_group_1 = leg_name.startswith('FL') or leg_name.startswith('RR')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        if phase < 0.25:
            # Phase 1: Rightward drift initiation
            # Group 1 (FL/RR) extends outward, Group 2 (FR/RL) tucks inward
            progress = phase / 0.25
            
            if is_group_1:
                # Extend outward (lateral positive for FL, negative for RR)
                lateral_sign = 1.0 if leg_name.startswith('FL') else -1.0
                foot[1] += lateral_sign * self.lateral_extension * progress
                foot[0] += self.forward_bias * progress if is_front else 0.0
            else:
                # Tuck inward
                lateral_sign = 1.0 if leg_name.startswith('FR') else -1.0
                foot[1] -= lateral_sign * self.lateral_extension * 0.3 * progress
        
        elif phase < 0.5:
            # Phase 2: Leftward drift transition
            # Group 2 (FR/RL) extends outward, Group 1 (FL/RR) tucks inward
            progress = (phase - 0.25) / 0.25
            
            if is_group_1:
                # Transition from extended to tucked
                lateral_sign = 1.0 if leg_name.startswith('FL') else -1.0
                extension = self.lateral_extension * (1.0 - progress)
                foot[1] += lateral_sign * extension
                foot[0] += self.forward_bias * (1.0 - progress) if is_front else 0.0
            else:
                # Transition from tucked to extended
                lateral_sign = 1.0 if leg_name.startswith('FR') else -1.0
                tuck_amount = 0.3 * (1.0 - progress)
                extend_amount = progress
                foot[1] += lateral_sign * (self.lateral_extension * extend_amount - self.lateral_extension * tuck_amount)
        
        elif phase < 0.75:
            # Phase 3: Synchronized circular paddling (peak yaw)
            # All legs execute circular motion with phase offsets
            paddling_progress = (phase - 0.5) / 0.25
            leg_phase_offset = self.paddling_phase_offsets[leg_name]
            angle = 2.0 * np.pi * (paddling_progress + leg_phase_offset)
            
            # Circular motion in body frame (lateral-vertical plane projected)
            foot[1] += self.paddling_radius * np.cos(angle) * (1.0 if is_front else -1.0)
            foot[0] += self.paddling_radius * np.sin(angle) * 0.5
        
        else:
            # Phase 4: Stabilization return to neutral
            # Smooth return from paddling to base position
            progress = (phase - 0.75) / 0.25
            decay = 1.0 - progress
            
            # Residual from paddling phase end
            paddling_progress = 1.0
            leg_phase_offset = self.paddling_phase_offsets[leg_name]
            angle = 2.0 * np.pi * (paddling_progress + leg_phase_offset)
            
            foot[1] += self.paddling_radius * np.cos(angle) * (1.0 if is_front else -1.0) * decay
            foot[0] += self.paddling_radius * np.sin(angle) * 0.5 * decay
        
        return foot