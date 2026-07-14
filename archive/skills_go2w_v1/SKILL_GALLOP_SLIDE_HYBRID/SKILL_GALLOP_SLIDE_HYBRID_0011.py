from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_GALLOP_SLIDE_HYBRID_MotionGenerator(BaseMotionGenerator):
    """
    Gallop-slide hybrid locomotion: explosive rear-leg thrust with flight phase,
    followed by extended low-friction glide on all four legs.
    
    Phase breakdown:
      [0.0, 0.15]: compression_gather - rear legs compress, front legs extend forward
      [0.15, 0.3]: thrust_launch - explosive rear extension, brief flight, landing transition
      [0.3, 0.6]: glide_extension - wide four-leg sliding stance, momentum preservation
      [0.6, 0.75]: inward_gather - legs retract inward
      [0.75, 1.0]: return_compression - return to compressed starting position
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # cycle frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Leg motion parameters
        self.compression_depth = 0.10  # how much rear legs compress vertically
        self.front_extension = 0.15  # how far forward front legs extend during gather
        self.glide_stance_width_forward = 0.20  # front leg forward extension during glide
        self.glide_stance_width_rear = 0.18  # rear leg rearward extension during glide
        self.glide_stance_lateral_spread = 0.05  # lateral widening during glide
        self.flight_lift_height = 0.08  # maximum height during swing phase
        
        # Base motion parameters
        self.thrust_vx_peak = 2.5  # peak forward velocity during thrust
        self.thrust_vz_peak = 1.2  # peak upward velocity during launch
        self.glide_vx_initial = 2.0  # initial forward velocity entering glide
        self.glide_vx_final = 0.4  # final forward velocity exiting glide
        self.compression_vz = -0.3  # downward velocity during compression
        
        # Pitch control parameters
        self.compression_pitch_rate = -0.8  # pitch down during compression
        self.launch_pitch_rate = 1.5  # pitch up during launch
        self.landing_pitch_rate = -1.2  # pitch down for landing
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        # Phase 0.0 - 0.15: compression_gather
        if phase < 0.15:
            phase_local = phase / 0.15
            # Decelerate horizontally as compression occurs
            vx = self.glide_vx_final * (1.0 - 0.5 * phase_local)
            vz = self.compression_vz * np.sin(np.pi * phase_local)
            pitch_rate = self.compression_pitch_rate * (1.0 - phase_local)
        
        # Phase 0.15 - 0.3: thrust_launch (includes flight and landing)
        elif phase < 0.3:
            phase_local = (phase - 0.15) / 0.15
            
            # Explosive acceleration in first half, then ballistic descent
            if phase_local < 0.5:  # thrust and early flight
                vx = self.thrust_vx_peak * (0.5 + phase_local)
                vz = self.thrust_vz_peak * np.sin(np.pi * phase_local * 2.0)
                pitch_rate = self.launch_pitch_rate * (1.0 - phase_local * 2.0)
            else:  # late flight and landing preparation
                vx = self.glide_vx_initial
                vz = -self.thrust_vz_peak * 0.5 * (phase_local - 0.5) * 2.0
                pitch_rate = self.landing_pitch_rate * ((phase_local - 0.5) * 2.0)
        
        # Phase 0.3 - 0.6: glide_extension
        elif phase < 0.6:
            phase_local = (phase - 0.3) / 0.3
            # Exponential velocity decay during glide
            decay = np.exp(-2.0 * phase_local)
            vx = self.glide_vx_initial * decay + self.glide_vx_final * (1.0 - decay)
            vz = 0.0  # maintain constant height
            pitch_rate = 0.0  # stable level glide
        
        # Phase 0.6 - 0.75: inward_gather
        elif phase < 0.75:
            phase_local = (phase - 0.6) / 0.15
            vx = self.glide_vx_final * (1.0 - 0.3 * phase_local)
            vz = -0.15 * np.sin(np.pi * phase_local)
            pitch_rate = 0.3 * np.sin(np.pi * phase_local)
        
        # Phase 0.75 - 1.0: return_compression
        else:
            phase_local = (phase - 0.75) / 0.25
            vx = self.glide_vx_final * 0.7 * (1.0 - phase_local)
            vz = self.compression_vz * np.sin(np.pi * phase_local)
            pitch_rate = self.compression_pitch_rate * 0.5 * phase_local
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_rear = leg_name.startswith('RL') or leg_name.startswith('RR')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Lateral offset during glide (positive for left legs, negative for right)
        lateral_sign = 1.0 if is_left else -1.0
        
        # Phase 0.0 - 0.15: compression_gather
        if phase < 0.15:
            phase_local = phase / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            
            if is_front:
                # Front legs extend forward
                foot[0] = base_pos[0] + self.front_extension * smooth
                foot[2] = base_pos[2]
            else:
                # Rear legs compress downward and shift forward
                foot[0] = base_pos[0] + 0.05 * smooth
                foot[2] = base_pos[2] - self.compression_depth * smooth
        
        # Phase 0.15 - 0.3: thrust_launch (swing phase)
        elif phase < 0.3:
            phase_local = (phase - 0.15) / 0.15
            
            if is_front:
                # Front legs transition from gathered to glide stance
                # Retract slightly then extend to wide forward position
                if phase_local < 0.5:
                    # Early flight: retract toward body
                    swing_progress = phase_local * 2.0
                    foot[0] = base_pos[0] + self.front_extension * (1.0 - 0.3 * swing_progress)
                    foot[2] = base_pos[2] + self.flight_lift_height * np.sin(np.pi * swing_progress)
                else:
                    # Landing preparation: extend to glide position
                    swing_progress = (phase_local - 0.5) * 2.0
                    foot[0] = base_pos[0] + self.front_extension * 0.7 + (self.glide_stance_width_forward - self.front_extension * 0.7) * swing_progress
                    foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread * swing_progress
                    foot[2] = base_pos[2] + self.flight_lift_height * np.sin(np.pi * (0.5 + swing_progress * 0.5))
            else:
                # Rear legs: explosive extension, flight, then to glide stance
                if phase_local < 0.3:
                    # Explosive thrust extension
                    thrust_progress = phase_local / 0.3
                    foot[0] = base_pos[0] + 0.05 - 0.08 * thrust_progress
                    foot[2] = base_pos[2] - self.compression_depth * (1.0 - thrust_progress)
                elif phase_local < 0.6:
                    # Early flight: lift off
                    flight_progress = (phase_local - 0.3) / 0.3
                    foot[0] = base_pos[0] - 0.03
                    foot[2] = base_pos[2] + self.flight_lift_height * np.sin(np.pi * flight_progress)
                else:
                    # Landing preparation: extend to rear glide position
                    landing_progress = (phase_local - 0.6) / 0.4
                    foot[0] = base_pos[0] - 0.03 - self.glide_stance_width_rear * landing_progress
                    foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread * landing_progress
                    foot[2] = base_pos[2] + self.flight_lift_height * (1.0 - landing_progress)
        
        # Phase 0.3 - 0.6: glide_extension (stance - sliding)
        elif phase < 0.6:
            phase_local = (phase - 0.3) / 0.3
            
            if is_front:
                # Wide forward stance, foot slides rearward relative to body
                foot[0] = base_pos[0] + self.glide_stance_width_forward - 0.3 * phase_local
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread
                foot[2] = base_pos[2]
            else:
                # Wide rear stance, foot slides forward relative to body
                foot[0] = base_pos[0] - self.glide_stance_width_rear + 0.25 * phase_local
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread
                foot[2] = base_pos[2]
        
        # Phase 0.6 - 0.75: inward_gather
        elif phase < 0.75:
            phase_local = (phase - 0.6) / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            
            if is_front:
                # Retract from glide position toward neutral
                glide_x = base_pos[0] + self.glide_stance_width_forward - 0.3
                target_x = base_pos[0] + 0.05
                foot[0] = glide_x + (target_x - glide_x) * smooth
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread * (1.0 - smooth)
                foot[2] = base_pos[2]
            else:
                # Retract from rear glide position toward neutral
                glide_x = base_pos[0] - self.glide_stance_width_rear + 0.25
                target_x = base_pos[0] + 0.03
                foot[0] = glide_x + (target_x - glide_x) * smooth
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread * (1.0 - smooth)
                foot[2] = base_pos[2] - 0.02 * smooth
        
        # Phase 0.75 - 1.0: return_compression
        else:
            phase_local = (phase - 0.75) / 0.25
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            
            if is_front:
                # Continue forward to gathered position
                foot[0] = base_pos[0] + 0.05 + (self.front_extension - 0.05) * smooth
                foot[2] = base_pos[2]
            else:
                # Compress and move slightly forward
                foot[0] = base_pos[0] + 0.03 + 0.02 * smooth
                foot[2] = base_pos[2] - 0.02 - (self.compression_depth - 0.02) * smooth
        
        return foot