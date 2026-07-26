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
        
        # Leg motion parameters (reduced to prevent joint limit violations and ground penetration)
        self.compression_depth = 0.04  # reduced from 0.10 to prevent excessive knee flexion
        self.front_extension = 0.12  # reduced from 0.15 for safer reach
        self.glide_stance_width_forward = 0.13  # reduced from 0.20 to stay within joint limits
        self.glide_stance_width_rear = 0.12  # reduced from 0.18 to stay within joint limits
        self.glide_stance_lateral_spread = 0.02  # reduced from 0.05 for safer lateral extension
        self.flight_lift_height = 0.06  # reduced from 0.08 for smoother transitions
        
        # Base motion parameters
        self.thrust_vx_peak = 2.2  # slightly reduced for smoother motion
        self.thrust_vz_peak = 0.9  # reduced from 1.2 for lower flight arc
        self.glide_vx_initial = 1.8  # reduced for smoother entry
        self.glide_vx_final = 0.4
        
        # Pitch control parameters (reduced for smoother transitions)
        self.compression_pitch_rate = -0.5  # reduced from -0.8
        self.launch_pitch_rate = 1.0  # reduced from 1.5
        self.landing_pitch_rate = -0.8  # reduced from -1.2
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Critical fix: eliminate downward velocity during compression to prevent ground penetration.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        # Phase 0.0 - 0.15: compression_gather
        # CRITICAL FIX: removed downward base velocity to prevent ground penetration
        if phase < 0.15:
            phase_local = phase / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            # Decelerate horizontally as compression occurs
            vx = self.glide_vx_final * (1.0 - 0.3 * smooth)
            vz = 0.0  # FIXED: was compression_vz * sin - now zero to prevent base sinking
            pitch_rate = self.compression_pitch_rate * smooth * (1.0 - smooth)  # smoother pitch
        
        # Phase 0.15 - 0.3: thrust_launch (includes flight and landing)
        elif phase < 0.3:
            phase_local = (phase - 0.15) / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            
            # Explosive acceleration in first half, then ballistic descent
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
            # Exponential velocity decay during glide
            decay = np.exp(-2.5 * phase_local)
            vx = self.glide_vx_initial * decay + self.glide_vx_final * (1.0 - decay)
            vz = 0.0  # maintain constant height during glide
            pitch_rate = 0.0  # stable level glide
        
        # Phase 0.6 - 0.75: inward_gather
        elif phase < 0.75:
            phase_local = (phase - 0.6) / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            vx = self.glide_vx_final * (1.0 - 0.2 * smooth)
            vz = 0.0  # keep level during gather
            pitch_rate = 0.2 * np.sin(2.0 * np.pi * phase_local)
        
        # Phase 0.75 - 1.0: return_compression
        else:
            phase_local = (phase - 0.75) / 0.25
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            vx = self.glide_vx_final * 0.8 * (1.0 - smooth)
            vz = 0.0  # FIXED: was downward velocity - now zero to prevent sinking
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
        All parameters reduced to prevent joint limit violations and ground penetration.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_rear = leg_name.startswith('RL') or leg_name.startswith('RR')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Lateral offset during glide
        lateral_sign = 1.0 if is_left else -1.0
        
        # Phase 0.0 - 0.15: compression_gather
        if phase < 0.15:
            phase_local = phase / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            
            if is_front:
                # Front legs extend forward, maintain ground contact
                foot[0] = base_pos[0] + self.front_extension * smooth
                foot[2] = base_pos[2]  # maintain height
            else:
                # Rear legs compress moderately (reduced depth to prevent violations)
                foot[0] = base_pos[0] + 0.03 * smooth
                foot[2] = base_pos[2] - self.compression_depth * smooth
        
        # Phase 0.15 - 0.3: thrust_launch (swing phase)
        elif phase < 0.3:
            phase_local = (phase - 0.15) / 0.15
            swing_smooth = np.sin(np.pi * phase_local)
            
            if is_front:
                # Front legs transition from gathered to glide stance
                if phase_local < 0.5:
                    # Early flight: slight retraction
                    progress = phase_local * 2.0
                    retract_smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
                    foot[0] = base_pos[0] + self.front_extension * (1.0 - 0.2 * retract_smooth)
                    foot[2] = base_pos[2] + self.flight_lift_height * swing_smooth
                else:
                    # Landing preparation: extend to glide position
                    progress = (phase_local - 0.5) * 2.0
                    extend_smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
                    foot[0] = base_pos[0] + self.front_extension * 0.8 + (self.glide_stance_width_forward - self.front_extension * 0.8) * extend_smooth
                    foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread * extend_smooth
                    foot[2] = base_pos[2] + self.flight_lift_height * np.sin(np.pi * (0.5 + progress * 0.5))
            else:
                # Rear legs: explosive extension, flight, then to glide stance
                if phase_local < 0.25:
                    # Explosive thrust extension
                    thrust_smooth = phase_local / 0.25
                    foot[0] = base_pos[0] + 0.03 * (1.0 - thrust_smooth) - 0.05 * thrust_smooth
                    foot[2] = base_pos[2] - self.compression_depth * (1.0 - thrust_smooth)
                elif phase_local < 0.5:
                    # Early flight: lift off
                    flight_progress = (phase_local - 0.25) / 0.25
                    foot[0] = base_pos[0] - 0.05
                    foot[2] = base_pos[2] + self.flight_lift_height * np.sin(np.pi * flight_progress)
                else:
                    # Landing preparation: extend to rear glide position
                    landing_progress = (phase_local - 0.5) / 0.5
                    land_smooth = 0.5 - 0.5 * np.cos(np.pi * landing_progress)
                    foot[0] = base_pos[0] - 0.05 - (self.glide_stance_width_rear - 0.05) * land_smooth
                    foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread * land_smooth
                    foot[2] = base_pos[2] + self.flight_lift_height * (1.0 - land_smooth)
        
        # Phase 0.3 - 0.6: glide_extension (stance - sliding)
        elif phase < 0.6:
            phase_local = (phase - 0.3) / 0.3
            slide_smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            
            if is_front:
                # Wide forward stance, foot slides rearward relative to body
                foot[0] = base_pos[0] + self.glide_stance_width_forward - 0.25 * slide_smooth
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread
                foot[2] = base_pos[2]
            else:
                # Wide rear stance, foot slides forward relative to body
                foot[0] = base_pos[0] - self.glide_stance_width_rear + 0.20 * slide_smooth
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread
                foot[2] = base_pos[2]
        
        # Phase 0.6 - 0.75: inward_gather
        elif phase < 0.75:
            phase_local = (phase - 0.6) / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            
            if is_front:
                # Retract from glide position toward neutral
                glide_x = base_pos[0] + self.glide_stance_width_forward - 0.25
                target_x = base_pos[0] + 0.04
                foot[0] = glide_x + (target_x - glide_x) * smooth
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread * (1.0 - smooth)
                foot[2] = base_pos[2]
            else:
                # Retract from rear glide position toward neutral
                glide_x = base_pos[0] - self.glide_stance_width_rear + 0.20
                target_x = base_pos[0] + 0.02
                foot[0] = glide_x + (target_x - glide_x) * smooth
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_lateral_spread * (1.0 - smooth)
                foot[2] = base_pos[2] - 0.01 * smooth
        
        # Phase 0.75 - 1.0: return_compression
        else:
            phase_local = (phase - 0.75) / 0.25
            smooth = 0.5 - 0.5 * np.cos(np.pi * phase_local)
            
            if is_front:
                # Continue forward to gathered position
                start_x = base_pos[0] + 0.04
                target_x = base_pos[0] + self.front_extension
                foot[0] = start_x + (target_x - start_x) * smooth
                foot[2] = base_pos[2]
            else:
                # Compress and move slightly forward
                start_x = base_pos[0] + 0.02
                target_x = base_pos[0] + 0.03
                foot[0] = start_x + (target_x - start_x) * smooth
                start_z = base_pos[2] - 0.01
                target_z = base_pos[2] - self.compression_depth
                foot[2] = start_z + (target_z - start_z) * smooth

        return foot