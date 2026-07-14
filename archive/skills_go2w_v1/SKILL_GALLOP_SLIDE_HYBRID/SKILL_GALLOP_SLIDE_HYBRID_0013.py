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
        
        # Leg motion parameters (final tuning to eliminate residual ground penetration)
        self.compression_depth = 0.027  # reduced from 0.04 to eliminate final 0.021m penetration
        self.front_extension = 0.12  # kept from previous successful iteration
        self.glide_stance_width_forward = 0.13  # kept - no violations in glide phase
        self.glide_stance_width_rear = 0.12  # kept - no violations in glide phase
        self.glide_stance_lateral_spread = 0.02  # kept - safe lateral extension
        self.flight_lift_height = 0.06  # kept - smooth transitions achieved
        
        # Base motion parameters
        self.thrust_vx_peak = 2.2
        self.thrust_vz_peak = 0.9
        self.glide_vx_initial = 1.8
        self.glide_vx_final = 0.4
        
        # Pitch control parameters
        self.compression_pitch_rate = -0.5
        self.launch_pitch_rate = 1.0
        self.landing_pitch_rate = -0.8
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Base vertical velocity kept at zero during compression phases to prevent ground penetration.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        # Phase 0.0 - 0.15: compression_gather
        if phase < 0.15:
            phase_local = phase / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            vx = self.glide_vx_final * (1.0 - 0.3 * smooth)
            vz = 0.0  # maintain zero to prevent base sinking
            pitch_rate = self.compression_pitch_rate * smooth * (1.0 - smooth)
        
        # Phase 0.15 - 0.3: thrust_launch
        elif phase < 0.3:
            phase_local = (phase - 0.15) / 0.15
            
            if phase_local < 0.4:  # thrust and early flight
                thrust_progress = phase_local / 0.4
                vx = self.glide_vx_final + (self.thrust_vx_peak - self.glide_vx_final) * thrust_progress
                vz = self.thrust_vz_peak * np.sin(np.pi * thrust_progress)
                pitch_rate = self.launch_pitch_rate * np.sin(np.pi * thrust_progress)
            else:  # late flight and landing preparation
                land_progress = (phase_local - 0.4) / 0.6
                vx = self.thrust_vx_peak * (1.0 - 0.2 * land_progress)
                vz = -self.thrust_vz_peak * 0.4 * land_progress
                pitch_rate = self.landing_pitch_rate * land_progress * 0.5
        
        # Phase 0.3 - 0.6: glide_extension
        elif phase < 0.6:
            phase_local = (phase - 0.3) / 0.3
            decay = np.exp(-2.5 * phase_local)
            vx = self.glide_vx_initial * decay + self.glide_vx_final * (1.0 - decay)
            vz = 0.0
            pitch_rate = 0.0
        
        # Phase 0.6 - 0.75: inward_gather
        elif phase < 0.75:
            phase_local = (phase - 0.6) / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            vx = self.glide_vx_final * (1.0 - 0.2 * smooth)
            vz = 0.0
            pitch_rate = 0.2 * np.sin(2.0 * np.pi * phase_local)
        
        # Phase 0.75 - 1.0: return_compression
        else:
            phase_local = (phase - 0.75) / 0.25
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            vx = self.glide_vx_final * 0.8 * (1.0 - smooth)
            vz = 0.0  # maintain zero during final compression
            pitch_rate = self.compression_pitch_rate * 0.3 * smooth
        
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
        Compression depth reduced to 0.027m to eliminate residual ground penetration.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_rear = leg_name.startswith('RL') or leg_name.startswith('RR')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        lateral_sign = 1.0 if is_left else -1.0
        
        # Phase 0.0 - 0.15: compression_gather
        if phase < 0.15:
            phase_local = phase / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            
            if is_front:
                foot[0] = base_pos[0] + self.front_extension * smooth
                foot[2] = base_pos[2]
            else:
                foot[0] = base_pos[0] + 0.03 * smooth
                foot[2] = base_pos[2] - self.compression_depth * smooth
        
        # Phase 0.15 - 0.3: thrust_launch
        elif phase < 0.3:
            phase_local = (phase - 0.15) / 0.15
            swing_smooth = np.sin(np.pi * phase_local)
            
            if is_front:
                if phase_local < 0.5:
                    progress = phase_local * 2.0
                    retract_smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
                    foot[0] = base_pos[0] + self.front_extension * (1.0 - 0.2 * retract_smooth)
                    foot[2] = base_pos[2] + self.flight_lift_height * swing_smooth
                else:
                    progress = (phase_local - 0.5) * 2.0
                    extend_smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
                    foot[0] = base_pos[0] + self.front_extension * 0.8 + (self.glide_stance_width_forward - self.front_extension * 0.8) * extend_smooth
                    foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread * extend_smooth
                    foot[2] = base_pos[2] + self.flight_lift_height * np.sin(np.pi * (0.5 + progress * 0.5))
            else:
                if phase_local < 0.25:
                    thrust_smooth = phase_local / 0.25
                    foot[0] = base_pos[0] + 0.03 * (1.0 - thrust_smooth) - 0.05 * thrust_smooth
                    foot[2] = base_pos[2] - self.compression_depth * (1.0 - thrust_smooth)
                elif phase_local < 0.5:
                    flight_progress = (phase_local - 0.25) / 0.25
                    foot[0] = base_pos[0] - 0.05
                    foot[2] = base_pos[2] + self.flight_lift_height * np.sin(np.pi * flight_progress)
                else:
                    landing_progress = (phase_local - 0.5) / 0.5
                    land_smooth = 0.5 - 0.5 * np.cos(np.pi * landing_progress)
                    foot[0] = base_pos[0] - 0.05 - (self.glide_stance_width_rear - 0.05) * land_smooth
                    foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread * land_smooth
                    foot[2] = base_pos[2] + self.flight_lift_height * (1.0 - land_smooth)
        
        # Phase 0.3 - 0.6: glide_extension
        elif phase < 0.6:
            phase_local = (phase - 0.3) / 0.3
            slide_smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            
            if is_front:
                foot[0] = base_pos[0] + self.glide_stance_width_forward - 0.25 * slide_smooth
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread
                foot[2] = base_pos[2]
            else:
                foot[0] = base_pos[0] - self.glide_stance_width_rear + 0.20 * slide_smooth
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread
                foot[2] = base_pos[2]
        
        # Phase 0.6 - 0.75: inward_gather
        elif phase < 0.75:
            phase_local = (phase - 0.6) / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            
            if is_front:
                glide_x = base_pos[0] + self.glide_stance_width_forward - 0.25
                target_x = base_pos[0] + 0.04
                foot[0] = glide_x + (target_x - glide_x) * smooth
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread * (1.0 - smooth)
                foot[2] = base_pos[2]
            else:
                glide_x = base_pos[0] - self.glide_stance_width_rear + 0.20
                target_x = base_pos[0] + 0.02
                foot[0] = glide_x + (target_x - glide_x) * smooth
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread * (1.0 - smooth)
                # Reduced intermediate compression from -0.01 to -0.007 for gentler gradient
                foot[2] = base_pos[2] - 0.007 * smooth
        
        # Phase 0.75 - 1.0: return_compression
        else:
            phase_local = (phase - 0.75) / 0.25
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            
            if is_front:
                start_x = base_pos[0] + 0.04
                target_x = base_pos[0] + self.front_extension
                foot[0] = start_x + (target_x - start_x) * smooth
                foot[2] = base_pos[2]
            else:
                start_x = base_pos[0] + 0.02
                target_x = base_pos[0] + 0.03
                foot[0] = start_x + (target_x - start_x) * smooth
                # Compression from -0.007 to final depth of -0.027 (reduced from -0.04)
                start_z = base_pos[2] - 0.007
                target_z = base_pos[2] - self.compression_depth
                foot[2] = start_z + (target_z - start_z) * smooth

        return foot