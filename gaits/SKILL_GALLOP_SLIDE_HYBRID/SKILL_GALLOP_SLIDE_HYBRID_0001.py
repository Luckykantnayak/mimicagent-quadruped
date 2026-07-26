from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_GALLOP_SLIDE_HYBRID_MotionGenerator(BaseMotionGenerator):
    """
    Hybrid gallop-slide locomotion skill.
    
    Phases:
      0.00-0.15: Compression/gathering - rear legs compress, front extend
      0.15-0.30: Thrust/launch - explosive rear extension with brief flight
      0.30-0.60: Glide extension - wide sliding stance, momentum conservation
      0.60-0.75: Regathering transition - legs pull inward
      0.75-1.00: Return to compression - complete gathering cycle
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.compression_depth = 0.12  # Vertical compression distance
        self.front_extension_x = 0.15  # Forward extension of front legs during compression
        self.rear_compression_x = 0.08  # Rearward pull of rear legs during compression
        
        self.glide_stance_width_front = 0.18  # Lateral extension during glide (front)
        self.glide_stance_width_rear = 0.16   # Lateral extension during glide (rear)
        self.glide_extension_x_front = 0.20   # Forward extension during glide (front)
        self.glide_extension_x_rear = 0.18    # Rearward extension during glide (rear)
        self.glide_stance_drop = 0.05         # Lower body during glide
        
        self.flight_height = 0.10  # Peak height during ballistic phase
        
        # Velocity parameters
        self.thrust_peak_vx = 1.8       # Peak forward velocity during thrust
        self.thrust_vz = 0.6            # Upward velocity during thrust
        self.glide_initial_vx = 1.2     # Initial glide velocity
        self.glide_final_vx = 0.3       # Final glide velocity (decay)
        
        # Pitch control parameters
        self.compression_pitch_rate = -0.8   # Nose-up during compression
        self.flight_pitch_rate = 1.2         # Nose-down during flight
        self.regather_pitch_rate = -0.6      # Nose-up during regathering
        
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
        vz = 0.0
        pitch_rate = 0.0
        
        if phase < 0.15:
            # Compression/gathering: decelerate, descend, pitch nose-up
            progress = phase / 0.15
            vx = self.glide_final_vx * (1.0 - progress)
            vz = -0.4 * np.sin(np.pi * progress)
            pitch_rate = self.compression_pitch_rate * (1.0 - progress)
            
        elif phase < 0.3:
            # Thrust/launch: explosive acceleration with ballistic arc
            progress = (phase - 0.15) / 0.15
            
            if progress < 0.5:
                # First half: acceleration phase
                vx = self.thrust_peak_vx * np.sin(np.pi * progress * 0.5)
                vz = self.thrust_vz * np.sin(np.pi * progress)
                pitch_rate = -self.compression_pitch_rate * (1.0 - 2.0 * progress)
            else:
                # Second half: ballistic descent
                vx = self.thrust_peak_vx * (1.0 - 0.3 * (progress - 0.5) * 2.0)
                vz = -self.thrust_vz * 0.5 * ((progress - 0.5) * 2.0)
                pitch_rate = self.flight_pitch_rate * ((progress - 0.5) * 2.0)
                
        elif phase < 0.6:
            # Glide extension: forward motion with gradual decay
            progress = (phase - 0.3) / 0.3
            vx = self.glide_initial_vx * (1.0 - progress) + self.glide_final_vx * progress
            vz = -0.1 * np.sin(np.pi * progress * 0.5)  # Slight downward for contact
            pitch_rate = 0.0
            
        elif phase < 0.75:
            # Regathering transition: continued deceleration, begin descent
            progress = (phase - 0.6) / 0.15
            vx = self.glide_final_vx * (1.0 - 0.3 * progress)
            vz = -0.3 * progress
            pitch_rate = self.regather_pitch_rate * progress
            
        else:
            # Return to compression: final deceleration and descent
            progress = (phase - 0.75) / 0.25
            vx = self.glide_final_vx * 0.7 * (1.0 - progress)
            vz = -0.3 * np.sin(np.pi * 0.5 + np.pi * 0.5 * progress)
            pitch_rate = self.regather_pitch_rate * (1.0 - 0.5 * progress)
        
        self.vel_world = np.array([vx, 0.0, vz])
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
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        lateral_sign = 1.0 if is_left else -1.0
        
        if phase < 0.15:
            # Compression/gathering phase
            progress = phase / 0.15
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            if is_front:
                # Front legs extend forward and slightly up
                foot[0] = base_pos[0] + self.front_extension_x * smooth_progress
                foot[2] = base_pos[2] - 0.02 * smooth_progress
            else:
                # Rear legs compress inward and down
                foot[0] = base_pos[0] - self.rear_compression_x * smooth_progress
                foot[2] = base_pos[2] + self.compression_depth * smooth_progress
            
        elif phase < 0.15 + 0.1:
            # Flight phase (0.15-0.25): maintain extended position
            if is_front:
                foot[0] = base_pos[0] + self.front_extension_x
                foot[2] = base_pos[2] - 0.02
            else:
                # Rear legs extend explosively then lift
                progress = (phase - 0.15) / 0.1
                foot[0] = base_pos[0] - self.rear_compression_x * (1.0 - progress) + self.glide_extension_x_rear * progress
                foot[2] = base_pos[2] + self.compression_depth * (1.0 - progress) + self.flight_height * np.sin(np.pi * progress)
                
        elif phase < 0.3:
            # Landing transition (0.25-0.3)
            progress = (phase - 0.25) / 0.05
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            if is_front:
                # Front legs transition to wide glide stance
                foot[0] = base_pos[0] + self.front_extension_x * (1.0 - smooth_progress) + self.glide_extension_x_front * smooth_progress
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_width_front * smooth_progress
                foot[2] = base_pos[2] - 0.02 - self.glide_stance_drop * smooth_progress
            else:
                # Rear legs land into wide glide stance
                foot[0] = base_pos[0] + self.glide_extension_x_rear
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_width_rear * smooth_progress
                foot[2] = base_pos[2] + self.flight_height * (1.0 - smooth_progress) - self.glide_stance_drop * smooth_progress
                
        elif phase < 0.6:
            # Glide extension: wide stable stance
            if is_front:
                foot[0] = base_pos[0] + self.glide_extension_x_front
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_width_front
                foot[2] = base_pos[2] - 0.02 - self.glide_stance_drop
            else:
                foot[0] = base_pos[0] + self.glide_extension_x_rear
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_width_rear
                foot[2] = base_pos[2] - self.glide_stance_drop
                
        elif phase < 0.75:
            # Regathering transition: legs pull inward
            progress = (phase - 0.6) / 0.15
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            if is_front:
                foot[0] = base_pos[0] + self.glide_extension_x_front * (1.0 - smooth_progress) + self.front_extension_x * smooth_progress
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_width_front * (1.0 - smooth_progress)
                foot[2] = base_pos[2] - 0.02 - self.glide_stance_drop * (1.0 - smooth_progress)
            else:
                foot[0] = base_pos[0] + self.glide_extension_x_rear * (1.0 - smooth_progress) - self.rear_compression_x * smooth_progress
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_width_rear * (1.0 - smooth_progress)
                foot[2] = base_pos[2] - self.glide_stance_drop * (1.0 - smooth_progress) + self.compression_depth * smooth_progress * 0.5
                
        else:
            # Return to compression: complete gathering
            progress = (phase - 0.75) / 0.25
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            if is_front:
                foot[0] = base_pos[0] + self.front_extension_x
                foot[2] = base_pos[2] - 0.02
            else:
                foot[0] = base_pos[0] - self.rear_compression_x
                foot[2] = base_pos[2] + self.compression_depth * (0.5 + 0.5 * smooth_progress)
        
        return foot