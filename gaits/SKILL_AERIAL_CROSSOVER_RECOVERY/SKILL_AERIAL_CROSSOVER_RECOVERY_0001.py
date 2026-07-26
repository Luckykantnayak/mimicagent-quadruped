from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_AERIAL_CROSSOVER_RECOVERY_MotionGenerator(BaseMotionGenerator):
    """
    Aerial Crossover Recovery Skill.
    
    The robot executes a dynamic skating maneuver where diagonal leg pairs
    alternate between aerial crossover waves and ground-supported skating.
    
    Key characteristics:
    - Diagonal pairs (FL-RR, FR-RL) alternate between aerial and stance
    - At least two feet always maintain ground contact
    - Crossover wave motion during aerial phases influences base yaw
    - Continuous forward skating propulsion throughout
    - Smooth transitions between aerial and stance phases
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions in BODY frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time
        self.t = 0.0
        
        # Base state
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Skating motion parameters
        self.forward_velocity = 0.8  # Base forward skating speed (m/s)
        self.aerial_height = 0.12  # Peak height during crossover (m)
        self.crossover_amplitude = 0.15  # Lateral crossover distance (m)
        
        # Base motion amplitudes
        self.base_z_lift_amp = 0.04  # Base height variation during aerial phases (m)
        self.yaw_rate_amp = 0.20  # Peak yaw rate during crossover (rad/s ~11 deg/s)
        self.roll_rate_amp = 0.15  # Peak roll rate for counterbalance (rad/s)
        self.lateral_drift_amp = 0.03  # Lateral velocity during aerial phases (m/s)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        Forward velocity is maintained throughout with slight modulation.
        Angular rates and lateral drift vary during aerial crossover phases.
        """
        
        # Forward velocity: constant with slight increase during double-support
        if (0.0 <= phase < 0.1) or (0.35 <= phase < 0.45) or (0.7 <= phase <= 1.0):
            vx = self.forward_velocity * 1.05  # Slight boost during double-support
        else:
            vx = self.forward_velocity
        
        # Lateral velocity and angular rates depend on which pair is aerial
        vy = 0.0
        vz = 0.0
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # FL-RR aerial crossover phase (0.1 - 0.35)
        if 0.1 <= phase < 0.35:
            local_phase = (phase - 0.1) / 0.25
            # Smooth envelope for aerial phase
            envelope = np.sin(np.pi * local_phase)
            
            # Rightward drift due to FL-RR momentum
            vy = self.lateral_drift_amp * envelope
            
            # Base lifts slightly
            vz = self.base_z_lift_amp * np.cos(2 * np.pi * local_phase)
            
            # Leftward yaw induced by crossover momentum
            yaw_rate = self.yaw_rate_amp * envelope
            
            # Left roll to counterbalance
            roll_rate = self.roll_rate_amp * envelope
        
        # Mid-transition (0.35 - 0.45): damping back to neutral
        elif 0.35 <= phase < 0.45:
            local_phase = (phase - 0.35) / 0.1
            damping = 1.0 - local_phase
            
            vy = self.lateral_drift_amp * damping * 0.5
            roll_rate = self.roll_rate_amp * damping * 0.3
            yaw_rate = self.yaw_rate_amp * damping * 0.3
        
        # FR-RL aerial crossover phase (0.45 - 0.7)
        elif 0.45 <= phase < 0.7:
            local_phase = (phase - 0.45) / 0.25
            envelope = np.sin(np.pi * local_phase)
            
            # Leftward drift due to FR-RL momentum
            vy = -self.lateral_drift_amp * envelope
            
            # Base lifts slightly
            vz = self.base_z_lift_amp * np.cos(2 * np.pi * local_phase)
            
            # Rightward yaw induced by crossover momentum
            yaw_rate = -self.yaw_rate_amp * envelope
            
            # Right roll to counterbalance
            roll_rate = -self.roll_rate_amp * envelope
        
        # Final transition (0.7 - 0.85): damping back to neutral
        elif 0.7 <= phase < 0.85:
            local_phase = (phase - 0.7) / 0.15
            damping = 1.0 - local_phase
            
            vy = -self.lateral_drift_amp * damping * 0.5
            roll_rate = -self.roll_rate_amp * damping * 0.3
            yaw_rate = -self.yaw_rate_amp * damping * 0.3
        
        # Cycle reset (0.85 - 1.0): neutral
        # (all rates already zero by default)
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
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
        Compute foot position in BODY frame for given leg and phase.
        
        FL and RR execute aerial crossover during phase 0.1-0.35.
        FR and RL execute aerial crossover during phase 0.45-0.7.
        All other phases are stance (skating contact).
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is front or rear for crossover direction
        is_front = leg_name.startswith('F')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # FL and RR aerial crossover (0.1 - 0.35)
        if (leg_name.startswith('FL') or leg_name.startswith('RR')) and (0.1 <= phase < 0.35):
            foot_pos = self._compute_crossover_trajectory(
                base_pos, phase, 0.1, 0.35, is_front, is_left
            )
        
        # FR and RL aerial crossover (0.45 - 0.7)
        elif (leg_name.startswith('FR') or leg_name.startswith('RL')) and (0.45 <= phase < 0.7):
            foot_pos = self._compute_crossover_trajectory(
                base_pos, phase, 0.45, 0.7, is_front, is_left
            )
        
        # All other phases: skating stance
        else:
            foot_pos = base_pos.copy()
            
            # During stance, foot drifts backward slightly relative to body
            # to simulate skating propulsion
            if 0.1 <= phase < 0.35:
                # FR and RL provide propulsion while FL-RR are aerial
                if leg_name.startswith('FR') or leg_name.startswith('RL'):
                    local_phase = (phase - 0.1) / 0.25
                    foot_pos[0] -= 0.05 * local_phase
            
            elif 0.45 <= phase < 0.7:
                # FL and RR provide propulsion while FR-RL are aerial
                if leg_name.startswith('FL') or leg_name.startswith('RR'):
                    local_phase = (phase - 0.45) / 0.25
                    foot_pos[0] -= 0.05 * local_phase
        
        return foot_pos

    def _compute_crossover_trajectory(self, base_pos, phase, phase_start, phase_end, is_front, is_left):
        """
        Compute crossover wave trajectory during aerial phase.
        
        The foot lifts, sweeps laterally inward (crossing body centerline),
        then arcs forward/rearward and outward, landing at nominal position.
        
        Args:
            base_pos: Nominal foot position in body frame
            phase: Current phase
            phase_start: Start of aerial phase
            phase_end: End of aerial phase
            is_front: True if front leg (FL/FR)
            is_left: True if left leg (FL/RL)
        
        Returns:
            foot_pos: 3D position in body frame
        """
        local_phase = (phase - phase_start) / (phase_end - phase_start)
        
        foot_pos = base_pos.copy()
        
        # Vertical trajectory: smooth liftoff, peak at mid-phase, smooth landing
        height_profile = np.sin(np.pi * local_phase)
        foot_pos[2] += self.aerial_height * height_profile
        
        # Lateral crossover trajectory: sweep inward then outward
        # Create a wave that crosses centerline
        crossover_profile = np.sin(2 * np.pi * local_phase - np.pi / 2)
        
        # Left legs cross rightward (positive y), right legs cross leftward (negative y)
        lateral_direction = -1.0 if is_left else 1.0
        foot_pos[1] += lateral_direction * self.crossover_amplitude * crossover_profile
        
        # Longitudinal motion: slight forward/rearward arc during crossover
        # Front legs arc forward, rear legs arc rearward
        longitudinal_direction = 1.0 if is_front else -1.0
        longitudinal_profile = 0.08 * np.sin(np.pi * local_phase)
        foot_pos[0] += longitudinal_direction * longitudinal_profile
        
        return foot_pos