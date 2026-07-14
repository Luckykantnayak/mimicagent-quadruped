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
        
        # Gait parameters - reduced for joint limit compliance
        self.step_length = 0.20
        self.step_height = 0.08
        self.body_compression = 0.03  # Used for body lowering, not foot displacement
        
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
        Vertical velocity coordinated with contact phases to prevent ground penetration.
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
            # Initial loading, all legs in contact - body begins to compress
            vx = self.forward_speed * 0.9
            vz = -0.02  # Gentle compression
            pitch_rate = 0.2
            yaw_rate = 0.1
            
        elif phase < 0.3:
            # Left legs pushing off - body rises
            vx = self.forward_speed * 1.1
            vy = -0.05
            vz = 0.12  # Upward launch
            roll_rate = -0.08
            pitch_rate = 0.3
            yaw_rate = 0.2
            
        elif phase < 0.45:
            # Peak flight phase, right legs pushing off - maintain altitude
            vx = self.forward_speed * 1.15
            vy = 0.08
            vz = 0.05  # Slight upward during flight
            roll_rate = 0.1
            pitch_rate = 0.0
            yaw_rate = self.yaw_rate_amplitude
            
        elif phase < 0.65:
            # Left legs landing - body begins descent but decelerating
            vx = self.forward_speed * 1.05
            vy = 0.1
            vz = -0.06  # Gentle descent
            roll_rate = -0.08
            pitch_rate = -0.25
            yaw_rate = self.yaw_rate_amplitude * 0.9
            
        elif phase < 0.8:
            # Right legs landing - stabilizing descent
            vx = self.forward_speed * 0.95
            vy = 0.05
            vz = -0.04  # Minimal descent as legs contact
            roll_rate = 0.08
            pitch_rate = -0.15
            yaw_rate = self.yaw_rate_amplitude * 0.6
            
        else:
            # Compression phase, all legs in contact - body settles
            vx = self.forward_speed
            vy = 0.0
            vz = 0.0  # No vertical motion during compression (feet stay on ground)
            roll_rate = 0.0
            pitch_rate = 0.1
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
        Feet maintain ground contact during stance; compression is handled by body lowering.
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
            swing_start = 0.15
            swing_end = 0.65
            
            if leg_phase < stance_end:
                # Initial stance: extend rearward, maintain ground contact
                progress = leg_phase / stance_end
                foot[0] -= self.step_length * (0.5 - progress)
                # No downward offset - foot stays on ground
                
            elif leg_phase < swing_end:
                # Swing phase: lift and move forward with asymmetric arc
                swing_progress = (leg_phase - swing_start) / (swing_end - swing_start)
                
                # Forward motion during swing
                foot[0] += self.step_length * (swing_progress - 0.5)
                
                # Asymmetric arc trajectory: peaks earlier to reduce joint demands
                # Use sin^0.8 to create earlier peak around 0.4 swing progress
                arc_height = np.sin(np.pi * swing_progress) ** 0.8
                foot[2] += self.step_height * arc_height
                
            else:
                # Final stance: hold forward position, maintain ground contact
                # Body compression is handled by base motion, not foot displacement
                foot[0] += self.step_length * 0.5
                
                # During compression phase (0.8-1.0), body lowers but feet stay on ground
                if leg_phase >= 0.8:
                    # Body has compressed, so feet appear relatively higher in body frame
                    compression_progress = (leg_phase - 0.8) / 0.2
                    foot[2] += self.body_compression * compression_progress
                
        else:
            # Right legs: stance [0.0-0.3] and [0.8-1.0], swing [0.3-0.8]
            stance_end = 0.3
            swing_start = 0.3
            swing_end = 0.8
            
            if leg_phase < stance_end:
                # Extended stance: extend rearward, maintain ground contact
                progress = leg_phase / stance_end
                foot[0] -= self.step_length * (0.5 - progress)
                # No downward offset - foot stays on ground
                
            elif leg_phase < swing_end:
                # Swing phase: lift and move forward (delayed)
                swing_progress = (leg_phase - swing_start) / (swing_end - swing_start)
                
                # Forward motion during swing
                foot[0] += self.step_length * (swing_progress - 0.5)
                
                # Asymmetric arc trajectory
                arc_height = np.sin(np.pi * swing_progress) ** 0.8
                foot[2] += self.step_height * arc_height
                
            else:
                # Compression stance: hold forward position, feet on ground
                foot[0] += self.step_length * 0.5
                
                # Body compression makes feet relatively higher in body frame
                compression_progress = (leg_phase - swing_end) / (1.0 - swing_end)
                foot[2] += self.body_compression * compression_progress
        
        return foot