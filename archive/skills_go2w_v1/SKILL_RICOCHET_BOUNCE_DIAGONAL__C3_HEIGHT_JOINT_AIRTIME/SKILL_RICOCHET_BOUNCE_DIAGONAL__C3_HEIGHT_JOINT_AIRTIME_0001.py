from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Ricochet bounce diagonal gait with alternating left/right diagonal bounces.
    
    Motion cycle:
    - [0.0-0.2]: Left compression with positive yaw
    - [0.2-0.4]: Right diagonal launch (aerial)
    - [0.4-0.6]: Right compression with negative yaw
    - [0.6-0.8]: Left diagonal launch (aerial)
    - [0.8-1.0]: Landing and yaw neutralization
    
    Base motion combines forward velocity with alternating lateral and vertical
    components, plus yaw oscillation to create zigzag ricochet pattern.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.compression_depth = 0.12  # Vertical compression during stance
        self.launch_height = 0.20  # Peak height during aerial phases
        self.lateral_amplitude = 0.15  # Lateral displacement for diagonal motion
        self.forward_velocity = 0.8  # Base forward speed
        self.yaw_amplitude = 0.52  # ~30 degrees yaw rotation (0.52 rad)
        
        # Leg extension parameters
        self.stance_retraction = 0.08  # Leg retraction during compression
        self.flight_extension = 0.10  # Leg extension during aerial phase
        self.lateral_foot_shift = 0.06  # Lateral foot adjustment during diagonal motion
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Creates zigzag pattern with yaw oscillation and diagonal launches.
        """
        vx = self.forward_velocity
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase 1: Left compression with positive yaw [0.0-0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            vz = -self.compression_depth * 3.0  # Downward compression
            yaw_rate = self.yaw_amplitude * 4.0  # Positive yaw rate
            
        # Phase 2: Right diagonal launch (aerial) [0.2-0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Upward then downward parabolic trajectory
            vz = self.launch_height * 5.0 * (1.0 - 2.0 * local_phase)
            vy = self.lateral_amplitude * 2.5  # Rightward velocity
            yaw_rate = 0.0  # No yaw during flight
            
        # Phase 3: Right compression with negative yaw [0.4-0.6]
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            vz = -self.compression_depth * 3.0  # Downward compression
            yaw_rate = -self.yaw_amplitude * 6.0  # Negative yaw rate (larger swing)
            vy = self.lateral_amplitude * 2.5 * (1.0 - local_phase)  # Decay rightward velocity
            
        # Phase 4: Left diagonal launch (aerial) [0.6-0.8]
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Upward then downward parabolic trajectory
            vz = self.launch_height * 5.0 * (1.0 - 2.0 * local_phase)
            vy = -self.lateral_amplitude * 2.5  # Leftward velocity
            yaw_rate = 0.0  # No yaw during flight
            
        # Phase 5: Landing and yaw neutralization [0.8-1.0]
        else:
            local_phase = (phase - 0.8) / 0.2
            vz = -self.compression_depth * 3.0 * (1.0 - local_phase)  # Compression decay
            yaw_rate = self.yaw_amplitude * 4.0  # Positive yaw to neutralize
            vy = -self.lateral_amplitude * 2.5 * (1.0 - local_phase)  # Decay leftward velocity
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
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
        Compute foot position in body frame based on phase and leg role.
        
        Left-side legs (FL, RL): Primary load during [0.0-0.2, 0.8-1.0]
        Right-side legs (FR, RR): Primary load during [0.4-0.6]
        All legs extend during aerial phases [0.2-0.4, 0.6-0.8]
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Phase 1: Left compression [0.0-0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            # Compression: legs retract, left legs load more
            foot[2] += self.compression_depth * local_phase
            if is_left:
                # Left legs retract more (primary loading)
                foot[0] -= self.stance_retraction * local_phase * (1.0 if is_front else -0.5)
            else:
                # Right legs retract less (secondary support)
                foot[0] -= self.stance_retraction * 0.5 * local_phase * (1.0 if is_front else -0.5)
                
        # Phase 2: Right diagonal launch (aerial) [0.2-0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # All legs extend symmetrically during flight
            foot[2] += self.compression_depth - self.flight_extension * local_phase
            # Legs extend forward/rearward and rightward
            foot[0] -= self.stance_retraction * (1.0 - local_phase) * (1.0 if is_front else -0.5)
            foot[0] += self.flight_extension * local_phase * (1.0 if is_front else -1.0)
            foot[1] += self.lateral_foot_shift * local_phase  # Rightward shift
            
        # Phase 3: Right compression [0.4-0.6]
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Compression: legs retract, right legs load more
            foot[2] += self.compression_depth * local_phase
            foot[1] += self.lateral_foot_shift * (1.0 - local_phase * 0.5)  # Maintain some rightward shift
            if not is_left:
                # Right legs retract more (primary loading)
                foot[0] += self.flight_extension * (1.0 - local_phase) * (1.0 if is_front else -1.0)
                foot[0] -= self.stance_retraction * local_phase * (1.0 if is_front else -0.5)
            else:
                # Left legs retract less (secondary support)
                foot[0] += self.flight_extension * (1.0 - local_phase) * (1.0 if is_front else -1.0)
                foot[0] -= self.stance_retraction * 0.5 * local_phase * (1.0 if is_front else -0.5)
                
        # Phase 4: Left diagonal launch (aerial) [0.6-0.8]
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # All legs extend symmetrically during flight
            foot[2] += self.compression_depth - self.flight_extension * local_phase
            # Legs extend forward/rearward and leftward
            foot[0] -= self.stance_retraction * (1.0 - local_phase) * (1.0 if is_front else -0.5)
            foot[0] += self.flight_extension * local_phase * (1.0 if is_front else -1.0)
            foot[1] += self.lateral_foot_shift * (1.0 - local_phase)  # Decay rightward shift
            foot[1] -= self.lateral_foot_shift * local_phase  # Add leftward shift
            
        # Phase 5: Landing and neutralization [0.8-1.0]
        else:
            local_phase = (phase - 0.8) / 0.2
            # Compression with transition to left-side loading
            foot[2] += self.compression_depth * (1.0 - local_phase * 0.5)
            foot[1] -= self.lateral_foot_shift * (1.0 - local_phase * 0.5)  # Decay leftward shift
            if is_left:
                # Left legs increase loading (preparing for next cycle)
                foot[0] += self.flight_extension * (1.0 - local_phase) * (1.0 if is_front else -1.0)
                foot[0] -= self.stance_retraction * local_phase * (1.0 if is_front else -0.5)
            else:
                # Right legs decrease loading
                foot[0] += self.flight_extension * (1.0 - local_phase) * (1.0 if is_front else -1.0)
                foot[0] -= self.stance_retraction * 0.3 * local_phase * (1.0 if is_front else -0.5)
        
        return foot