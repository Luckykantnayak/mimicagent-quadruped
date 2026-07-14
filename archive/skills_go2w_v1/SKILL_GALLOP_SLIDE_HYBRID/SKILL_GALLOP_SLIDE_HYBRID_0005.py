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
        self.freq = 0.6  # ~1.67 second cycle for gallop-slide
        
        # Base foot positions (BODY frame, neutral stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.compression_height = 0.12  # How much base lowers during compression
        self.thrust_vx_peak = 2.5  # Peak forward velocity during thrust
        self.thrust_vz_impulse = 0.8  # Upward velocity component during launch
        self.slide_initial_vx = 2.0  # Initial slide velocity
        self.slide_decay_rate = 3.0  # Velocity decay during slide
        
        # Leg motion parameters
        self.front_extension_x = 0.25  # Forward extension of front legs
        self.rear_compression_z = 0.15  # Rear leg compression amount
        self.rear_extension_x = -0.2  # Rearward extension during slide
        self.slide_stance_width = 0.15  # Lateral extension during slide
        self.flight_clearance = 0.1  # Leg lift during flight phase
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Internal tracking for smooth velocity transitions
        self.current_vx = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        Kinematic prescription:
        - Compression (0.0-0.15): slight forward, downward velocity, nose-down pitch
        - Thrust (0.15-0.3): explosive forward/upward velocity, nose-up pitch
        - Slide (0.3-0.6): decaying forward velocity, level pitch
        - Gather (0.6-0.75): continued deceleration
        - Return (0.75-1.0): minimal forward, downward velocity, nose-down pitch
        """
        
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        # Phase 0.0-0.15: Compression/gather
        if phase < 0.15:
            local_phase = phase / 0.15
            vx = 0.2 * local_phase  # Slight forward motion
            vz = -0.8 * np.sin(np.pi * local_phase)  # Downward compression
            pitch_rate = -0.3  # Nose tilts down
            
        # Phase 0.15-0.3: Thrust/launch
        elif phase < 0.3:
            local_phase = (phase - 0.15) / 0.15
            # Explosive acceleration with smooth profile
            accel_profile = np.sin(np.pi * local_phase)
            vx = self.thrust_vx_peak * accel_profile
            # Upward impulse early in thrust, then ballistic
            if local_phase < 0.5:
                vz = self.thrust_vz_impulse * np.sin(2 * np.pi * local_phase)
            else:
                vz = 0.0  # Ballistic phase
            pitch_rate = 1.5 * (1.0 - local_phase)  # Nose pitches up during launch
            
        # Phase 0.3-0.6: Slide/glide
        elif phase < 0.6:
            local_phase = (phase - 0.3) / 0.3
            # Exponential velocity decay simulating friction
            vx = self.slide_initial_vx * np.exp(-self.slide_decay_rate * local_phase)
            vz = 0.0  # Maintain height during slide
            # Return pitch to level at start of slide
            if local_phase < 0.2:
                pitch_rate = -1.0  # Nose returns to level
            else:
                pitch_rate = 0.0
                
        # Phase 0.6-0.75: Gather inward
        elif phase < 0.75:
            local_phase = (phase - 0.6) / 0.15
            # Continued deceleration
            vx = 0.5 * (1.0 - local_phase)
            vz = -0.2 * local_phase  # Begin lowering
            pitch_rate = 0.0
            
        # Phase 0.75-1.0: Return to compression
        else:
            local_phase = (phase - 0.75) / 0.25
            vx = 0.1 * (1.0 - local_phase)  # Minimal forward motion
            vz = -0.6 * np.sin(np.pi * local_phase)  # Return to compressed height
            pitch_rate = -0.4 * local_phase  # Nose tilts down for next cycle
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
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
        
        Front legs (FL, FR): extend forward during compression, lift during thrust, 
                              extend wide during slide, gather inward
        Rear legs (RL, RR): compress during gather, extend explosively during thrust,
                             extend wide during slide, gather inward
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        is_front = leg_name.startswith('F')
        is_left = leg_name.endswith('L')
        
        # Lateral offset for left vs right
        lateral_sign = 1.0 if is_left else -1.0
        
        # Phase 0.0-0.15: Compression/gather
        if phase < 0.15:
            local_phase = phase / 0.15
            if is_front:
                # Front legs extend forward and down
                foot[0] += self.front_extension_x * local_phase
                foot[2] -= 0.03 * local_phase
            else:
                # Rear legs compress (move up in body frame as joints flex)
                foot[0] += 0.05 * local_phase  # Slight forward
                foot[2] += self.rear_compression_z * local_phase
                
        # Phase 0.15-0.3: Thrust/launch
        elif phase < 0.3:
            local_phase = (phase - 0.15) / 0.15
            if is_front:
                # Front legs swing forward-upward during flight
                foot[0] += self.front_extension_x * (1.0 + 0.5 * local_phase)
                # Lift during flight (peak around mid-phase)
                if local_phase > 0.5:
                    flight_local = (local_phase - 0.5) / 0.5
                    foot[2] += self.flight_clearance * np.sin(np.pi * flight_local)
            else:
                # Rear legs explosively extend then lift
                extension_progress = min(local_phase / 0.6, 1.0)  # Extend through first 60%
                foot[0] += 0.05 - 0.1 * extension_progress  # Move rearward
                foot[2] += self.rear_compression_z * (1.0 - extension_progress)  # Extend downward
                # Lift during flight phase (last 40%)
                if local_phase > 0.6:
                    flight_local = (local_phase - 0.6) / 0.4
                    foot[2] += self.flight_clearance * np.sin(np.pi * flight_local)
                    
        # Phase 0.3-0.6: Slide/glide
        elif phase < 0.6:
            local_phase = (phase - 0.3) / 0.3
            # Smooth transition into wide stance at start, then hold
            stance_progress = min(local_phase / 0.2, 1.0)
            if is_front:
                # Front legs fully extended forward and outward
                foot[0] += self.front_extension_x * 1.5
                foot[1] += lateral_sign * self.slide_stance_width * stance_progress
                foot[2] -= 0.03
            else:
                # Rear legs fully extended rearward and outward
                foot[0] += self.rear_extension_x
                foot[1] += lateral_sign * self.slide_stance_width * stance_progress
                foot[2] -= 0.05
                
        # Phase 0.6-0.75: Gather inward
        elif phase < 0.75:
            local_phase = (phase - 0.6) / 0.15
            if is_front:
                # Retract from wide stance
                foot[0] += self.front_extension_x * 1.5 * (1.0 - local_phase)
                foot[1] += lateral_sign * self.slide_stance_width * (1.0 - local_phase)
                foot[2] -= 0.03 * (1.0 - local_phase)
            else:
                # Retract from wide stance
                foot[0] += self.rear_extension_x * (1.0 - local_phase)
                foot[1] += lateral_sign * self.slide_stance_width * (1.0 - local_phase)
                foot[2] -= 0.05 * (1.0 - local_phase)
                
        # Phase 0.75-1.0: Return to compression
        else:
            local_phase = (phase - 0.75) / 0.25
            if is_front:
                # Return to extended forward position for next compression
                foot[0] += self.front_extension_x * local_phase
                foot[2] -= 0.03 * local_phase
            else:
                # Return to compressed position
                foot[0] += 0.05 * local_phase
                foot[2] += self.rear_compression_z * local_phase
        
        return foot