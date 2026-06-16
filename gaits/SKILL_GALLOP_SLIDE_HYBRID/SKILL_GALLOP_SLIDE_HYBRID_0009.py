from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_GALLOP_SLIDE_HYBRID_MotionGenerator(BaseMotionGenerator):
    """
    Hybrid gallop-slide locomotion skill.
    
    Motion cycle:
    - Phase 0.0-0.15: Compression/gather (rear legs compress, front legs extend forward)
    - Phase 0.15-0.3: Thrust/launch (rear legs explosive extension, brief flight phase)
    - Phase 0.3-0.6: Slide/glide (all four legs extended wide, gliding on momentum)
    - Phase 0.6-0.75: Gather inward (legs retract from wide stance)
    - Phase 0.75-1.0: Return to compression (body lowers, preparing for next cycle)
    
    Base motion uses kinematic velocity commands to simulate gallop thrust and momentum-conserving slide.
    Leg trajectories are in BODY frame.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.6
        
        # Base foot positions (BODY frame, neutral stance) - use original positions without modification
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.thrust_vx_peak = 2.0
        self.thrust_vz_impulse = 0.5
        self.slide_initial_vx = 1.6
        self.slide_decay_rate = 2.2
        
        # Leg motion parameters
        self.front_extension_x = 0.18
        self.front_extension_x_compression = 0.16
        self.rear_compression_z = 0.03
        self.rear_compression_x = 0.10
        self.rear_extension_x = -0.15
        self.slide_stance_width_front = 0.12
        self.slide_stance_width_rear = 0.13
        self.flight_clearance = 0.035
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.nominal_height = 0.35

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        NO downward vz during compression - height control via foot placement only.
        """
        
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        # Phase 0.0-0.15: Compression/gather
        if phase < 0.15:
            local_phase = phase / 0.15
            vx = 0.12 * local_phase
            vz = 0.0
            pitch_rate = -0.15 * np.sin(np.pi * local_phase)
            
        # Phase 0.15-0.3: Thrust/launch (reduced pitch rate to ease rear leg joint demands)
        elif phase < 0.3:
            local_phase = (phase - 0.15) / 0.15
            accel_profile = np.sin(np.pi * local_phase)
            vx = self.thrust_vx_peak * accel_profile
            if local_phase < 0.5:
                vz = self.thrust_vz_impulse * np.sin(2 * np.pi * local_phase)
            else:
                vz = 0.0
            pitch_rate = 0.8 * np.sin(np.pi * local_phase)
            
        # Phase 0.3-0.6: Slide/glide
        elif phase < 0.6:
            local_phase = (phase - 0.3) / 0.3
            vx = self.slide_initial_vx * np.exp(-self.slide_decay_rate * local_phase)
            vz = 0.0
            if local_phase < 0.3:
                pitch_rate = -0.7 * (1.0 - np.cos(np.pi * local_phase / 0.3)) / 2.0
            else:
                pitch_rate = 0.0
                
        # Phase 0.6-0.75: Gather inward
        elif phase < 0.75:
            local_phase = (phase - 0.6) / 0.15
            vx = 0.35 * (1.0 - local_phase)
            vz = 0.0
            pitch_rate = 0.0
            
        # Phase 0.75-1.0: Return to compression
        else:
            local_phase = (phase - 0.75) / 0.25
            vx = 0.08 * (1.0 - local_phase)
            vz = 0.0
            pitch_rate = -0.2 * np.sin(np.pi * local_phase)
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        if self.root_pos[2] < self.nominal_height:
            self.root_pos[2] = self.nominal_height

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in BODY frame with corrected Z offsets (less negative = closer to body origin = less penetration).
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        is_front = leg_name.startswith('F')
        is_left = leg_name.endswith('L')
        
        lateral_sign = 1.0 if is_left else -1.0
        
        # Phase 0.0-0.15: Compression/gather
        if phase < 0.15:
            local_phase = phase / 0.15
            smooth = (1.0 - np.cos(np.pi * local_phase)) / 2.0
            if is_front:
                foot[0] += self.front_extension_x_compression * smooth
                foot[2] -= 0.04 * smooth
            else:
                foot[0] += self.rear_compression_x * smooth
                foot[2] += self.rear_compression_z * smooth
                
        # Phase 0.15-0.3: Thrust/launch
        elif phase < 0.3:
            local_phase = (phase - 0.15) / 0.15
            if is_front:
                smooth_profile = (1.0 - np.cos(np.pi * local_phase)) / 2.0
                extension = 1.0 + 0.2 * smooth_profile
                foot[0] += self.front_extension_x * extension
                if local_phase > 0.2:
                    flight_local = (local_phase - 0.2) / 0.8
                    foot[2] += self.flight_clearance * np.sin(np.pi * flight_local) - 0.04 * (1.0 - flight_local)
                else:
                    foot[2] -= 0.04
            else:
                if local_phase < 0.6:
                    # Extension phase with smooth blending preparation
                    extension_progress = local_phase / 0.6
                    smooth_ext = (1.0 - np.cos(np.pi * extension_progress)) / 2.0
                    foot[0] += self.rear_compression_x * (1.0 - smooth_ext) - 0.08 * smooth_ext
                    foot[2] += self.rear_compression_z * (1.0 - smooth_ext) - 0.04 * smooth_ext
                elif local_phase < 0.65:
                    # Smooth transition zone into flight
                    blend = (local_phase - 0.6) / 0.05
                    extension_z = -0.04
                    flight_z_start = self.flight_clearance * 0.0 - 0.02
                    foot[0] -= 0.08
                    foot[2] += extension_z * (1.0 - blend) + flight_z_start * blend
                else:
                    # Flight phase with simplified trajectory
                    flight_local = (local_phase - 0.65) / 0.35
                    foot[0] -= 0.08
                    foot[2] += self.flight_clearance * np.sin(np.pi * flight_local) - 0.02
                    
        # Phase 0.3-0.6: Slide/glide
        elif phase < 0.6:
            local_phase = (phase - 0.3) / 0.3
            if local_phase < 0.40:
                landing = local_phase / 0.40
                smooth_land = ((1.0 - np.cos(np.pi * landing)) / 2.0) ** 1.5
                if is_front:
                    foot[0] += self.front_extension_x * (1.2 + 0.15 * smooth_land)
                    foot[1] += lateral_sign * self.slide_stance_width_front * smooth_land
                    flight_z = self.flight_clearance * (1.0 - landing)
                    foot[2] += flight_z - 0.05 * smooth_land
                else:
                    foot[0] += self.rear_extension_x * smooth_land - 0.08 * (1.0 - smooth_land)
                    foot[1] += lateral_sign * self.slide_stance_width_rear * smooth_land
                    flight_z = self.flight_clearance * (1.0 - landing)
                    foot[2] += flight_z - 0.06 * smooth_land
            else:
                if is_front:
                    foot[0] += self.front_extension_x * 1.20
                    foot[1] += lateral_sign * self.slide_stance_width_front
                    foot[2] -= 0.05
                else:
                    foot[0] += self.rear_extension_x
                    foot[1] += lateral_sign * self.slide_stance_width_rear
                    foot[2] -= 0.06
                
        # Phase 0.6-0.75: Gather inward
        elif phase < 0.75:
            local_phase = (phase - 0.6) / 0.15
            smooth = (1.0 + np.cos(np.pi * local_phase)) / 2.0
            if is_front:
                foot[0] += self.front_extension_x * 1.20 * smooth
                foot[1] += lateral_sign * self.slide_stance_width_front * smooth
                foot[2] -= 0.05 * smooth
            else:
                foot[0] += self.rear_extension_x * smooth
                foot[1] += lateral_sign * self.slide_stance_width_rear * smooth
                foot[2] -= 0.06 * smooth
                
        # Phase 0.75-1.0: Return to compression
        else:
            local_phase = (phase - 0.75) / 0.25
            smooth = (1.0 - np.cos(np.pi * local_phase)) / 2.0
            if is_front:
                foot[0] += self.front_extension_x_compression * smooth
                foot[2] -= 0.04 * smooth
            else:
                foot[0] += self.rear_compression_x * smooth
                foot[2] += self.rear_compression_z * smooth
        
        return foot