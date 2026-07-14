from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_GALLOP_SLIDE_HYBRID_MotionGenerator(BaseMotionGenerator):
    """
    Hybrid gallop-slide locomotion skill.
    
    Phases:
      0.00-0.15: Compression/gathering - rear legs compress, front extend
      0.15-0.30: Thrust/launch - explosive rear extension with brief flight
      0.30-0.65: Glide extension - wide sliding stance, momentum conservation
      0.65-0.80: Regathering transition - legs pull inward
      0.80-1.00: Return to compression - complete gathering cycle
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - scaled for kinematic feasibility
        self.front_extension_x = 0.10  # Forward extension of front legs
        self.rear_compression_x = 0.05  # Rearward pull of rear legs during compression
        
        self.glide_stance_width_front = 0.09  # Lateral extension during glide
        self.glide_stance_width_rear = 0.07   # Lateral extension during glide (reduced for workspace)
        self.glide_extension_x_front = 0.13   # Forward extension during glide
        self.glide_extension_x_rear = 0.09    # Rearward extension during glide (reduced for workspace)
        self.glide_stance_lift = 0.018        # Increased minimum lift for safety margin
        
        self.flight_height = 0.08  # Peak height during ballistic phase
        
        # Velocity parameters
        self.thrust_peak_vx = 1.5       # Peak forward velocity during thrust
        self.thrust_vz = 0.5            # Upward velocity during thrust
        self.glide_initial_vx = 1.0     # Initial glide velocity
        self.glide_final_vx = 0.35      # Final glide velocity (decay)
        
        # Pitch control parameters - reduced magnitudes during ground contact phases
        self.compression_pitch_rate = -0.25   # Reduced nose-up during compression
        self.flight_pitch_rate = 0.8          # Nose-down during flight (airborne, safe)
        self.regather_pitch_rate = -0.2       # Reduced nose-up during regathering
        
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
            # Compression/gathering: decelerate, base descends, pitch nose-up
            progress = phase / 0.15
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            vx = self.glide_final_vx * (1.0 - smooth_progress)
            vz = -0.20 * np.sin(np.pi * progress)  # Base descends to create compression visual
            pitch_rate = self.compression_pitch_rate * np.sin(np.pi * progress)
            
        elif phase < 0.3:
            # Thrust/launch: explosive acceleration with ballistic arc
            progress = (phase - 0.15) / 0.15
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            if progress < 0.5:
                # First half: acceleration phase
                accel_progress = progress * 2.0
                vx = self.thrust_peak_vx * (0.5 * (1.0 - np.cos(np.pi * accel_progress * 0.5)))
                vz = self.thrust_vz * np.sin(np.pi * accel_progress * 0.5)
                pitch_rate = self.compression_pitch_rate * (1.0 - accel_progress)
            else:
                # Second half: ballistic descent
                descent_progress = (progress - 0.5) * 2.0
                vx = self.thrust_peak_vx * (1.0 - 0.2 * descent_progress)
                vz = -self.thrust_vz * 0.35 * descent_progress
                pitch_rate = self.flight_pitch_rate * descent_progress * 0.5
                
        elif phase < 0.65:
            # Glide extension: forward motion with gradual decay, minimal vertical
            progress = (phase - 0.3) / 0.35
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            vx = self.glide_initial_vx * (1.0 - smooth_progress) + self.glide_final_vx * smooth_progress
            vz = -0.05 * np.sin(np.pi * progress)
            pitch_rate = 0.0
            
        elif phase < 0.80:
            # Regathering transition: continued deceleration
            progress = (phase - 0.65) / 0.15
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            vx = self.glide_final_vx * (1.0 - 0.2 * smooth_progress)
            vz = -0.12 * smooth_progress
            pitch_rate = self.regather_pitch_rate * smooth_progress
            
        else:
            # Return to compression: final deceleration, base descends
            progress = (phase - 0.80) / 0.20
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            vx = self.glide_final_vx * 0.8 * (1.0 - smooth_progress)
            vz = -0.15 * np.sin(np.pi * smooth_progress)
            pitch_rate = self.regather_pitch_rate * (1.0 - 0.5 * smooth_progress)
        
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def get_pitch_compensation(self, foot_x_body):
        """
        Compute Z-offset needed to compensate for base pitch angle.
        Rear legs positioned behind base center (negative X) need positive Z-offset
        when base pitches nose-up (negative pitch angle).
        """
        # Extract pitch angle from quaternion
        qw, qx, qy, qz = self.root_quat
        pitch = np.arcsin(2.0 * (qw * qy - qz * qx))
        
        # Z compensation: rear feet need to rise in body frame when nose pitches up
        # foot_x_body is negative for rear legs, pitch is negative for nose-up
        # Product is positive, raising rear feet in body frame to maintain ground contact in world frame
        z_compensation = -foot_x_body * np.sin(pitch)
        
        return z_compensation

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        Includes pitch compensation to maintain ground contact in world frame.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        lateral_sign = 1.0 if is_left else -1.0
        
        if phase < 0.15:
            # Compression/gathering phase: rear legs pull back, front extend
            progress = phase / 0.15
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            if is_front:
                # Front legs extend forward slightly at ground level
                foot[0] = base_pos[0] + self.front_extension_x * smooth_progress
                foot[2] = base_pos[2] + self.glide_stance_lift
            else:
                # Rear legs compress rearward
                foot[0] = base_pos[0] - self.rear_compression_x * smooth_progress
                foot[2] = base_pos[2] + self.glide_stance_lift
                # Apply pitch compensation for rear legs
                foot[2] += self.get_pitch_compensation(foot[0])
            
        elif phase < 0.15 + 0.10:
            # Flight phase (0.15-0.25): maintain extended position with lift
            flight_progress = (phase - 0.15) / 0.10
            smooth_flight = np.sin(np.pi * flight_progress)
            
            if is_front:
                foot[0] = base_pos[0] + self.front_extension_x
                foot[2] = base_pos[2] + self.flight_height * smooth_flight * 0.5  # Front lifts less
            else:
                # Rear legs extend explosively then lift
                foot[0] = base_pos[0] - self.rear_compression_x * (1.0 - flight_progress) + self.glide_extension_x_rear * flight_progress
                foot[2] = base_pos[2] + self.flight_height * smooth_flight
                
        elif phase < 0.35:
            # Extended landing transition (0.25-0.35)
            progress = (phase - 0.25) / 0.10
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            if is_front:
                # Front legs transition to wide glide stance
                foot[0] = base_pos[0] + self.front_extension_x * (1.0 - smooth_progress) + self.glide_extension_x_front * smooth_progress
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_width_front * smooth_progress
                # Smooth touchdown from flight to ground
                flight_z = base_pos[2] + self.flight_height * 0.5 * (1.0 - progress)
                glide_z = base_pos[2] + self.glide_stance_lift
                foot[2] = flight_z * (1.0 - smooth_progress) + glide_z * smooth_progress
            else:
                # Rear legs land into wide glide stance
                foot[0] = base_pos[0] + self.glide_extension_x_rear
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_width_rear * smooth_progress
                # Smooth touchdown from flight height to ground
                flight_z = base_pos[2] + self.flight_height * np.sin(np.pi * (1.0 - progress))
                glide_z = base_pos[2] + self.glide_stance_lift
                foot[2] = flight_z * (1.0 - smooth_progress) + glide_z * smooth_progress
                # Apply pitch compensation during landing
                if smooth_progress > 0.5:
                    foot[2] += self.get_pitch_compensation(foot[0]) * (smooth_progress - 0.5) * 2.0
                
        elif phase < 0.65:
            # Glide extension: wide stable stance at ground level
            if is_front:
                foot[0] = base_pos[0] + self.glide_extension_x_front
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_width_front
                foot[2] = base_pos[2] + self.glide_stance_lift
            else:
                foot[0] = base_pos[0] + self.glide_extension_x_rear
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_width_rear
                foot[2] = base_pos[2] + self.glide_stance_lift
                # Apply pitch compensation for rear legs
                foot[2] += self.get_pitch_compensation(foot[0])
                
        elif phase < 0.80:
            # Regathering transition: legs pull inward
            progress = (phase - 0.65) / 0.15
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            if is_front:
                foot[0] = base_pos[0] + self.glide_extension_x_front * (1.0 - smooth_progress) + self.front_extension_x * smooth_progress
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_width_front * (1.0 - smooth_progress)
                foot[2] = base_pos[2] + self.glide_stance_lift
            else:
                # Rear legs regather: transition from extended glide to compressed rearward position
                foot[0] = base_pos[0] + self.glide_extension_x_rear * (1.0 - smooth_progress) - self.rear_compression_x * smooth_progress
                foot[1] = base_pos[1] + lateral_sign * self.glide_stance_width_rear * (1.0 - smooth_progress)
                foot[2] = base_pos[2] + self.glide_stance_lift
                # Apply pitch compensation for rear legs
                foot[2] += self.get_pitch_compensation(foot[0])
                
        else:
            # Return to compression: complete gathering
            progress = (phase - 0.80) / 0.20
            smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            
            if is_front:
                foot[0] = base_pos[0] + self.front_extension_x
                foot[2] = base_pos[2] + self.glide_stance_lift
            else:
                # Rear legs remain in compressed rearward position
                foot[0] = base_pos[0] - self.rear_compression_x
                foot[2] = base_pos[2] + self.glide_stance_lift
                # Apply pitch compensation for rear legs
                foot[2] += self.get_pitch_compensation(foot[0])
        
        return foot