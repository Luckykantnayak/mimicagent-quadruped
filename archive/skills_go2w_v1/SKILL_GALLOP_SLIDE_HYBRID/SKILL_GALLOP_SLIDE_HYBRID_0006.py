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
        
        # Base foot positions (BODY frame, neutral stance) - lowered to prevent penetration
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            pos = v.copy()
            pos[2] -= 0.08  # Lower all feet in body frame to account for compression phases
            self.base_feet_pos_body[k] = pos
        
        # Motion parameters - reduced magnitudes to stay within joint limits
        self.compression_height = 0.06  # Reduced from 0.12
        self.thrust_vx_peak = 2.2  # Slightly reduced
        self.thrust_vz_impulse = 0.6  # Reduced from 0.8
        self.slide_initial_vx = 1.8  # Reduced from 2.0
        self.slide_decay_rate = 2.5  # Slightly reduced decay
        
        # Leg motion parameters - moderated to avoid joint limits
        self.front_extension_x = 0.20  # Reduced from 0.25
        self.rear_compression_z = 0.06  # Reduced from 0.15 to avoid knee overflexion
        self.rear_extension_x = -0.15  # Reduced from -0.2
        self.slide_stance_width = 0.10  # Reduced from 0.15
        self.flight_clearance = 0.06  # Reduced from 0.1
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.current_vx = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Reduced vertical velocities to prevent ground penetration.
        """
        
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        # Phase 0.0-0.15: Compression/gather
        if phase < 0.15:
            local_phase = phase / 0.15
            vx = 0.15 * local_phase
            vz = -0.3 * np.sin(np.pi * local_phase)  # Reduced from -0.8
            pitch_rate = -0.2 * np.sin(np.pi * local_phase)  # Smoothed
            
        # Phase 0.15-0.3: Thrust/launch
        elif phase < 0.3:
            local_phase = (phase - 0.15) / 0.15
            accel_profile = np.sin(np.pi * local_phase)
            vx = self.thrust_vx_peak * accel_profile
            if local_phase < 0.5:
                vz = self.thrust_vz_impulse * np.sin(2 * np.pi * local_phase)
            else:
                vz = 0.05  # Slight upward to maintain flight
            pitch_rate = 1.2 * np.sin(np.pi * local_phase)  # Smoothed
            
        # Phase 0.3-0.6: Slide/glide
        elif phase < 0.6:
            local_phase = (phase - 0.3) / 0.3
            vx = self.slide_initial_vx * np.exp(-self.slide_decay_rate * local_phase)
            vz = 0.0
            # Smooth pitch return to level
            if local_phase < 0.3:
                pitch_rate = -0.8 * (1.0 - np.cos(np.pi * local_phase / 0.3)) / 2.0
            else:
                pitch_rate = 0.0
                
        # Phase 0.6-0.75: Gather inward
        elif phase < 0.75:
            local_phase = (phase - 0.6) / 0.15
            vx = 0.4 * (1.0 - local_phase)
            vz = -0.15 * np.sin(np.pi * local_phase)  # Gentle lowering
            pitch_rate = 0.0
            
        # Phase 0.75-1.0: Return to compression
        else:
            local_phase = (phase - 0.75) / 0.25
            vx = 0.1 * (1.0 - local_phase)
            vz = -0.25 * np.sin(np.pi * local_phase)  # Reduced from -0.6
            pitch_rate = -0.3 * np.sin(np.pi * local_phase)  # Smoothed
        
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
        Compute foot position in BODY frame with reduced offsets to avoid joint limits.
        Smooth transitions to avoid velocity spikes.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        is_front = leg_name.startswith('F')
        is_left = leg_name.endswith('L')
        
        lateral_sign = 1.0 if is_left else -1.0
        
        # Phase 0.0-0.15: Compression/gather
        if phase < 0.15:
            local_phase = phase / 0.15
            smooth = (1.0 - np.cos(np.pi * local_phase)) / 2.0  # Smooth interpolation
            if is_front:
                foot[0] += self.front_extension_x * smooth
                foot[2] -= 0.02 * smooth  # Slight lowering
            else:
                foot[0] += 0.03 * smooth
                foot[2] += self.rear_compression_z * smooth  # Reduced compression
                
        # Phase 0.15-0.3: Thrust/launch
        elif phase < 0.3:
            local_phase = (phase - 0.15) / 0.15
            if is_front:
                # Smooth extension forward and upward during flight
                extension = 1.0 + 0.3 * local_phase
                foot[0] += self.front_extension_x * extension
                # Smooth lift during flight
                if local_phase > 0.4:
                    flight_local = (local_phase - 0.4) / 0.6
                    foot[2] += self.flight_clearance * np.sin(np.pi * flight_local)
                else:
                    foot[2] -= 0.02
            else:
                # Gradual extension with smooth flight transition
                if local_phase < 0.7:
                    extension_progress = local_phase / 0.7
                    smooth_ext = (1.0 - np.cos(np.pi * extension_progress)) / 2.0
                    foot[0] += 0.03 * (1.0 - smooth_ext) - 0.05 * smooth_ext
                    foot[2] += self.rear_compression_z * (1.0 - smooth_ext)
                else:
                    # Flight phase
                    flight_local = (local_phase - 0.7) / 0.3
                    foot[0] -= 0.05
                    foot[2] += self.flight_clearance * np.sin(np.pi * flight_local)
                    
        # Phase 0.3-0.6: Slide/glide - extended transition for smooth landing
        elif phase < 0.6:
            local_phase = (phase - 0.3) / 0.3
            # Extended landing transition
            if local_phase < 0.25:
                landing = local_phase / 0.25
                smooth_land = (1.0 - np.cos(np.pi * landing)) / 2.0
                if is_front:
                    foot[0] += self.front_extension_x * (1.3 + 0.2 * smooth_land)
                    foot[1] += lateral_sign * self.slide_stance_width * smooth_land
                    # Smooth descent from flight
                    flight_z = self.flight_clearance * (1.0 - landing)
                    foot[2] += flight_z - 0.02 * smooth_land
                else:
                    foot[0] += self.rear_extension_x * smooth_land - 0.05 * (1.0 - smooth_land)
                    foot[1] += lateral_sign * self.slide_stance_width * smooth_land
                    flight_z = self.flight_clearance * (1.0 - landing)
                    foot[2] += flight_z - 0.03 * smooth_land
            else:
                # Stable slide stance
                if is_front:
                    foot[0] += self.front_extension_x * 1.5
                    foot[1] += lateral_sign * self.slide_stance_width
                    foot[2] -= 0.02
                else:
                    foot[0] += self.rear_extension_x
                    foot[1] += lateral_sign * self.slide_stance_width
                    foot[2] -= 0.03
                
        # Phase 0.6-0.75: Gather inward
        elif phase < 0.75:
            local_phase = (phase - 0.6) / 0.15
            smooth = (1.0 + np.cos(np.pi * local_phase)) / 2.0  # Smooth retraction
            if is_front:
                foot[0] += self.front_extension_x * 1.5 * smooth
                foot[1] += lateral_sign * self.slide_stance_width * smooth
                foot[2] -= 0.02 * smooth
            else:
                foot[0] += self.rear_extension_x * smooth
                foot[1] += lateral_sign * self.slide_stance_width * smooth
                foot[2] -= 0.03 * smooth
                
        # Phase 0.75-1.0: Return to compression
        else:
            local_phase = (phase - 0.75) / 0.25
            smooth = (1.0 - np.cos(np.pi * local_phase)) / 2.0
            if is_front:
                foot[0] += self.front_extension_x * smooth
                foot[2] -= 0.02 * smooth
            else:
                foot[0] += 0.03 * smooth
                foot[2] += self.rear_compression_z * smooth
        
        return foot