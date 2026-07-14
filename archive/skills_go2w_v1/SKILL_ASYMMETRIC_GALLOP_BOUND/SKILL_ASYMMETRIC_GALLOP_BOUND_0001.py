from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ASYMMETRIC_GALLOP_BOUND_MotionGenerator(BaseMotionGenerator):
    """
    Asymmetric bounding gait with left-side legs leading right-side legs by phase offset.
    
    Left legs (FL, RL) push off and land first, right legs (FR, RR) follow with 0.15 phase delay.
    This asymmetry generates rightward yaw while bounding forward.
    
    Key features:
    - Synchronized left pair (FL+RL) and right pair (FR+RR)
    - 0.15 phase offset between pairs creates lateral asymmetry
    - Brief flight phase when all legs airborne (~0.3-0.45)
    - Alternating diagonal support during landing phases
    - Compression phase with all legs in contact (0.8-1.0)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.2
        
        # Base foot positions in BODY frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Gait parameters
        self.step_length = 0.25
        self.step_height = 0.12
        self.compression_depth = 0.04
        
        # Phase offsets: left legs (FL, RL) at 0.0, right legs (FR, RR) at 0.15
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RL'):
                self.phase_offsets[leg] = 0.0
            elif leg.startswith('FR') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.15
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters
        self.forward_speed = 1.2
        self.yaw_rate_amplitude = 0.8

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands to create asymmetric bounding motion with rightward curve.
        
        Phase breakdown:
        [0.0-0.3]: Left pushoff - forward acceleration, upward velocity, begin rightward yaw
        [0.3-0.45]: Peak flight - maintain forward, peak upward, increasing yaw rate
        [0.45-0.65]: Left landing - forward maintained, descending, sustained yaw
        [0.65-0.8]: Right landing - slight deceleration, settling, yaw stabilizing
        [0.8-1.0]: Compression - constant forward, downward compression, minimal yaw
        """
        
        # Linear velocity components
        vx = self.forward_speed
        vy = 0.0
        vz = 0.0
        
        # Angular velocity components
        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        
        # Phase-dependent velocity modulation
        if phase < 0.15:
            # Initial loading, all legs in contact
            vx = self.forward_speed * 0.9
            vz = 0.02
            pitch_rate = 0.3
            yaw_rate = 0.1
            
        elif phase < 0.3:
            # Left legs pushing off
            vx = self.forward_speed * 1.1
            vy = -0.05
            vz = 0.15
            roll_rate = -0.1
            pitch_rate = 0.5
            yaw_rate = 0.2
            
        elif phase < 0.45:
            # Peak flight phase, right legs pushing off
            vx = self.forward_speed * 1.15
            vy = 0.08
            vz = 0.1
            roll_rate = 0.15
            pitch_rate = 0.0
            yaw_rate = self.yaw_rate_amplitude
            
        elif phase < 0.65:
            # Left legs landing, right legs in flight
            vx = self.forward_speed * 1.05
            vy = 0.1
            vz = -0.12
            roll_rate = -0.1
            pitch_rate = -0.4
            yaw_rate = self.yaw_rate_amplitude * 0.9
            
        elif phase < 0.8:
            # Right legs landing, stabilizing
            vx = self.forward_speed * 0.95
            vy = 0.05
            vz = -0.08
            roll_rate = 0.1
            pitch_rate = -0.2
            yaw_rate = self.yaw_rate_amplitude * 0.6
            
        else:
            # Compression phase, all legs in contact
            vx = self.forward_speed
            vy = 0.0
            vz = -0.05
            roll_rate = 0.0
            pitch_rate = 0.15
            yaw_rate = 0.2
        
        # Set velocity commands
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
        
        Left legs (FL, RL): phase offset 0.0
          - [0.0-0.15]: stance, extending rearward
          - [0.15-0.65]: swing, lifting and moving forward
          - [0.65-1.0]: stance, supporting and compressing
        
        Right legs (FR, RR): phase offset 0.15
          - [0.0-0.3]: stance, extending rearward (delayed)
          - [0.3-0.8]: swing, lifting and moving forward (delayed)
          - [0.8-1.0]: stance, supporting and compressing
        """
        
        # Apply leg-specific phase offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Get base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if left or right leg
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        if is_left:
            # Left legs: stance [0.0-0.15] and [0.65-1.0], swing [0.15-0.65]
            stance_end = 0.15
            swing_end = 0.65
            
            if leg_phase < stance_end:
                # Initial stance: extend rearward
                progress = leg_phase / stance_end
                foot[0] -= self.step_length * (0.5 - progress)
                foot[2] -= self.compression_depth * (1.0 - progress)
                
            elif leg_phase < swing_end:
                # Swing phase: lift and move forward
                swing_progress = (leg_phase - stance_end) / (swing_end - stance_end)
                
                # Forward motion during swing
                foot[0] += self.step_length * (swing_progress - 0.5)
                
                # Arc trajectory: sin for smooth lift and descent
                arc_angle = np.pi * swing_progress
                foot[2] += self.step_height * np.sin(arc_angle)
                
            else:
                # Final stance: compress and hold
                stance_progress = (leg_phase - swing_end) / (1.0 - swing_end)
                foot[0] += self.step_length * 0.5
                foot[2] -= self.compression_depth * stance_progress
                
        else:
            # Right legs: stance [0.0-0.3] and [0.8-1.0], swing [0.3-0.8]
            stance_end = 0.3
            swing_end = 0.8
            
            if leg_phase < stance_end:
                # Extended stance: extend rearward (delayed pushoff)
                progress = leg_phase / stance_end
                foot[0] -= self.step_length * (0.5 - progress)
                foot[2] -= self.compression_depth * (1.0 - progress)
                
            elif leg_phase < swing_end:
                # Swing phase: lift and move forward (delayed)
                swing_progress = (leg_phase - stance_end) / (swing_end - stance_end)
                
                # Forward motion during swing
                foot[0] += self.step_length * (swing_progress - 0.5)
                
                # Arc trajectory
                arc_angle = np.pi * swing_progress
                foot[2] += self.step_height * np.sin(arc_angle)
                
            else:
                # Compression stance: compress and hold
                stance_progress = (leg_phase - swing_end) / (1.0 - swing_end)
                foot[0] += self.step_length * 0.5
                foot[2] -= self.compression_depth * stance_progress
        
        return foot