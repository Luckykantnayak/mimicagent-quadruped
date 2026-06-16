from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_INCHWORM_COMPRESSION_MotionGenerator(BaseMotionGenerator):
    """
    Inchworm locomotion gait with cyclic compression and extension.
    
    Phase structure:
      [0.0, 0.3]: compression - rear legs swing forward, base lowers and pitches down
      [0.3, 0.5]: compressed hold - all feet grounded, base stationary
      [0.5, 0.8]: extension - front legs swing forward, base rises and pitches up
      [0.8, 1.0]: extended hold - all feet grounded, base stationary
    
    Foot trajectories in BODY frame.
    Base motion prescribed via velocity commands in WORLD frame.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for deliberate inchworm motion
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Swing parameters
        self.compression_step_length = 0.15  # Rear legs swing forward during compression
        self.extension_step_length = 0.20    # Front legs swing forward during extension
        self.swing_height = 0.05             # Low swing for ground-hugging motion
        
        # Base motion parameters
        self.compression_vx = 0.05           # Slight forward drift during compression
        self.compression_vz = -0.08          # Base descends during compression
        self.compression_pitch_rate = -0.3   # Nose tilts down
        
        self.extension_vx = 0.15             # Larger forward motion during extension
        self.extension_vz = 0.08             # Base rises during extension
        self.extension_pitch_rate = 0.3      # Nose tilts up
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent velocity commands.
        
        Phase [0.0, 0.3]: compression - forward drift, downward, pitch down
        Phase [0.3, 0.5]: compressed hold - zero velocity
        Phase [0.5, 0.8]: extension - forward, upward, pitch up
        Phase [0.8, 1.0]: extended hold - zero velocity
        """
        
        if phase < 0.3:
            # Compression phase
            progress = phase / 0.3
            smooth = 0.5 * (1.0 - np.cos(np.pi * progress))  # Smooth interpolation
            vx = self.compression_vx * np.sin(np.pi * progress)
            vz = self.compression_vz * np.sin(np.pi * progress)
            pitch_rate = self.compression_pitch_rate * np.sin(np.pi * progress)
            
        elif phase < 0.5:
            # Compressed hold phase
            vx = 0.0
            vz = 0.0
            pitch_rate = 0.0
            
        elif phase < 0.8:
            # Extension phase
            progress = (phase - 0.5) / 0.3
            smooth = 0.5 * (1.0 - np.cos(np.pi * progress))
            vx = self.extension_vx * np.sin(np.pi * progress)
            vz = self.extension_vz * np.sin(np.pi * progress)
            pitch_rate = self.extension_pitch_rate * np.sin(np.pi * progress)
            
        else:
            # Extended hold phase
            vx = 0.0
            vz = 0.0
            pitch_rate = 0.0
        
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
        Compute foot position in BODY frame for given phase.
        
        Front legs (FL, FR):
          - Stance during [0.0, 0.5]
          - Swing forward during [0.5, 0.8]
          - Stance during [0.8, 1.0]
          
        Rear legs (RL, RR):
          - Swing forward during [0.0, 0.3]
          - Stance during [0.3, 1.0]
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        if leg_name.startswith('FL') or leg_name.startswith('FR'):
            # Front legs
            if phase < 0.5:
                # Stance phase during compression and hold
                # Foot appears to move backward in body frame as base advances
                progress = phase / 0.5
                foot[0] -= self.extension_step_length * progress * 0.5
                
            elif phase < 0.8:
                # Swing phase during extension
                swing_progress = (phase - 0.5) / 0.3
                
                # Forward swing with arc
                foot[0] += self.extension_step_length * (swing_progress - 0.5)
                
                # Swing height arc
                swing_angle = np.pi * swing_progress
                foot[2] += self.swing_height * np.sin(swing_angle)
                
            else:
                # Stance phase during extended hold
                foot[0] += self.extension_step_length * 0.3
                
        else:
            # Rear legs (RL, RR)
            if phase < 0.3:
                # Swing phase during compression
                swing_progress = phase / 0.3
                
                # Forward swing with arc
                foot[0] += self.compression_step_length * (swing_progress - 0.5)
                
                # Swing height arc
                swing_angle = np.pi * swing_progress
                foot[2] += self.swing_height * np.sin(swing_angle)
                
            elif phase < 0.5:
                # Stance during compressed hold
                foot[0] += self.compression_step_length * 0.2
                
            elif phase < 0.8:
                # Stance phase during extension
                # Foot appears to move backward in body frame as base extends
                progress = (phase - 0.5) / 0.3
                foot[0] += self.compression_step_length * 0.2
                foot[0] -= self.extension_step_length * progress * 0.8
                
            else:
                # Stance during extended hold
                foot[0] -= self.extension_step_length * 0.4
        
        return foot