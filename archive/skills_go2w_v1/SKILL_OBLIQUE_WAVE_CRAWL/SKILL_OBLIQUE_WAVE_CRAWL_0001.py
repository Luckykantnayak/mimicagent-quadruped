from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_OBLIQUE_WAVE_CRAWL_MotionGenerator(BaseMotionGenerator):
    """
    Oblique wave crawl gait with diagonal body wave propagation from RL to FR.
    
    - All four feet maintain continuous ground contact throughout the cycle
    - Body wave creates diagonal thrust at ~45 degrees to longitudinal axis
    - Wave initiation at RL (phase 0.0-0.25), mid-body (0.25-0.5), peak at FR (0.5-0.75), completion (0.75-1.0)
    - Base motion driven by integrated velocity commands in world frame
    - Leg trajectories in body frame create wave-like undulation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Wave frequency (Hz) - slow crawl for stability
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Wave propagation timing offsets (RL -> mid-body -> FR)
        # RL leads, FR lags by 0.5, FL and RR are intermediate
        self.wave_phase_offsets = {
            leg_names[0]: 0.15 if leg_names[0].startswith('FL') else 0.0,   # FL
            leg_names[1]: 0.55 if leg_names[1].startswith('FR') else 0.0,   # FR
            leg_names[2]: 0.0 if leg_names[2].startswith('RL') else 0.0,    # RL
            leg_names[3]: 0.30 if leg_names[3].startswith('RR') else 0.0,   # RR
        }
        
        # Ensure correct assignment regardless of leg_names order
        for leg in leg_names:
            if leg.startswith('FL'):
                self.wave_phase_offsets[leg] = 0.15
            elif leg.startswith('FR'):
                self.wave_phase_offsets[leg] = 0.55
            elif leg.startswith('RL'):
                self.wave_phase_offsets[leg] = 0.0
            elif leg.startswith('RR'):
                self.wave_phase_offsets[leg] = 0.30
        
        # Leg motion parameters
        self.stance_displacement = 0.18  # Max fore-aft displacement in body frame
        self.lateral_displacement = 0.12  # Lateral shift during wave
        self.vertical_variation = 0.04  # Minimal vertical motion (continuous contact)
        
        # Base velocity parameters (world frame)
        self.vx_base = 0.15  # Base forward velocity amplitude
        self.vy_base = 0.12  # Base lateral velocity amplitude (diagonal component)
        
        # Angular velocity parameters (world frame)
        self.roll_rate_amp = 0.3  # Body roll amplitude (rad/s)
        self.pitch_rate_amp = 0.2  # Body pitch amplitude (rad/s)
        self.yaw_rate_amp = 0.15  # Body yaw amplitude (rad/s)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base velocities based on wave phase.
        Wave propagates RL -> FR creating diagonal motion with body undulation.
        """
        
        # Linear velocity commands (world frame)
        # Phase 0.0-0.25: Wave initiation at RL, building diagonal velocity
        # Phase 0.25-0.5: Mid-body wave, sustained diagonal motion
        # Phase 0.5-0.75: Wave peak at FR, maximum diagonal velocity
        # Phase 0.75-1.0: Wave completion, deceleration to neutral
        
        if phase < 0.25:
            # Wave initiation - velocity building
            progress = phase / 0.25
            vx = self.vx_base * progress
            vy = self.vy_base * progress * 0.5  # Slight left initially
        elif phase < 0.5:
            # Mid-body wave - transition to full diagonal
            progress = (phase - 0.25) / 0.25
            vx = self.vx_base
            vy = self.vy_base * (0.5 + 0.5 * progress)  # Increasing right component
        elif phase < 0.75:
            # Wave peak - maximum diagonal velocity
            vx = self.vx_base
            vy = self.vy_base
        else:
            # Wave completion - smooth deceleration
            progress = (phase - 0.75) / 0.25
            vx = self.vx_base * (1.0 - progress)
            vy = self.vy_base * (1.0 - progress)
        
        # Angular velocity commands (world frame)
        # Roll: negative (left) during RL push, positive (right) during FR push
        # Pitch: positive (rear compress) early, negative (front compress) late
        # Yaw: positive throughout to maintain diagonal trajectory
        
        if phase < 0.25:
            # RL initiation - left roll, rear compression, building yaw
            roll_rate = -self.roll_rate_amp * np.sin(np.pi * phase / 0.25)
            pitch_rate = self.pitch_rate_amp * np.sin(np.pi * phase / 0.25)
            yaw_rate = self.yaw_rate_amp * (phase / 0.25)
        elif phase < 0.5:
            # Mid-body - roll transition, pitch neutral, sustained yaw
            progress = (phase - 0.25) / 0.25
            roll_rate = self.roll_rate_amp * np.sin(np.pi * progress)
            pitch_rate = self.pitch_rate_amp * (1.0 - 2.0 * progress)
            yaw_rate = self.yaw_rate_amp
        elif phase < 0.75:
            # FR peak - right roll, front compression, peak yaw
            progress = (phase - 0.5) / 0.25
            roll_rate = self.roll_rate_amp * np.sin(np.pi * (0.5 + 0.5 * progress))
            pitch_rate = -self.pitch_rate_amp * np.sin(np.pi * progress)
            yaw_rate = self.yaw_rate_amp
        else:
            # Wave completion - all rates return to zero
            progress = (phase - 0.75) / 0.25
            roll_rate = self.roll_rate_amp * (1.0 - progress) * 0.5
            pitch_rate = -self.pitch_rate_amp * (1.0 - progress) * 0.5
            yaw_rate = self.yaw_rate_amp * (1.0 - progress)
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, 0.0])
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
        Compute foot position in body frame based on wave propagation.
        All feet maintain contact - trajectories are smooth stance motions.
        """
        
        # Get leg-specific wave phase
        leg_phase = (phase + self.wave_phase_offsets[leg_name]) % 1.0
        
        # Base position for this leg
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg role in wave propagation
        is_RL = leg_name.startswith('RL')
        is_FR = leg_name.startswith('FR')
        is_FL = leg_name.startswith('FL')
        is_RR = leg_name.startswith('RR')
        
        # Wave-based stance trajectory
        # During high-load phase: foot pushes rearward in body frame
        # During low-load phase: foot repositions forward in body frame
        
        if is_RL:
            # RL: Wave initiator (0.0-0.25 high load, then reset)
            if leg_phase < 0.25:
                # Active push phase
                progress = leg_phase / 0.25
                foot[0] += self.stance_displacement * (0.5 - progress)
                foot[1] -= self.lateral_displacement * np.sin(np.pi * progress)
                foot[2] += self.vertical_variation * np.sin(2 * np.pi * progress)
            else:
                # Reset phase
                progress = (leg_phase - 0.25) / 0.75
                foot[0] += self.stance_displacement * (-0.5 + progress * 0.5)
                foot[1] -= self.lateral_displacement * 0.1 * np.sin(np.pi * progress)
                
        elif is_FR:
            # FR: Wave terminator (0.5-0.75 equivalent to 0.0-0.25 with offset)
            if leg_phase < 0.3:
                # Pre-load phase
                progress = leg_phase / 0.3
                foot[0] += self.stance_displacement * (0.3 - 0.1 * progress)
                foot[1] += self.lateral_displacement * 0.1 * progress
            elif leg_phase < 0.55:
                # Active push phase
                progress = (leg_phase - 0.3) / 0.25
                foot[0] += self.stance_displacement * (0.2 - progress * 0.7)
                foot[1] += self.lateral_displacement * np.sin(np.pi * progress)
                foot[2] += self.vertical_variation * np.sin(2 * np.pi * progress)
            else:
                # Reset phase
                progress = (leg_phase - 0.55) / 0.45
                foot[0] += self.stance_displacement * (-0.5 + progress * 0.8)
                foot[1] += self.lateral_displacement * 0.1 * (1.0 - progress)
                
        elif is_FL:
            # FL: Counter-diagonal support (0.25-0.5 high load)
            if leg_phase < 0.15:
                # Pre-transition
                progress = leg_phase / 0.15
                foot[0] += self.stance_displacement * (0.4 * (1.0 - progress))
                foot[1] -= self.lateral_displacement * 0.05
            elif leg_phase < 0.5:
                # Active support and push
                progress = (leg_phase - 0.15) / 0.35
                foot[0] += self.stance_displacement * (0.3 - progress * 0.8)
                foot[1] -= self.lateral_displacement * 0.2 * np.sin(np.pi * progress)
                foot[2] += self.vertical_variation * 0.5 * np.sin(2 * np.pi * progress)
            else:
                # Reset phase
                progress = (leg_phase - 0.5) / 0.5
                foot[0] += self.stance_displacement * (-0.5 + progress * 0.9)
                foot[1] -= self.lateral_displacement * 0.05 * (1.0 - progress)
                
        elif is_RR:
            # RR: Counter-diagonal support (0.25-0.5 moderate load)
            if leg_phase < 0.3:
                # Pre-load
                progress = leg_phase / 0.3
                foot[0] += self.stance_displacement * (0.2 * (1.0 - progress))
            elif leg_phase < 0.6:
                # Active support
                progress = (leg_phase - 0.3) / 0.3
                foot[0] += self.stance_displacement * (0.1 - progress * 0.6)
                foot[1] += self.lateral_displacement * 0.15 * np.sin(np.pi * progress)
                foot[2] += self.vertical_variation * 0.3 * np.sin(2 * np.pi * progress)
            else:
                # Reset phase
                progress = (leg_phase - 0.6) / 0.4
                foot[0] += self.stance_displacement * (-0.5 + progress * 0.7)
                foot[1] += self.lateral_displacement * 0.05 * (1.0 - progress)
        
        return foot