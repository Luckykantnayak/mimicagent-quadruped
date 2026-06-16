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
        self.freq = 0.5
        
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        self.compression_step_length = 0.15
        self.extension_step_length = 0.20
        self.swing_height = 0.07  # Increased for better clearance
        
        self.compression_vx = 0.05
        self.compression_vz = -0.06  # Reduced magnitude to limit descent
        self.compression_pitch_rate = -0.3
        
        self.extension_vx = 0.15
        self.extension_vz = 0.06  # Matched to compression magnitude
        self.extension_pitch_rate = 0.3
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Precompute integrated base motion for compensation
        # Compression phase: 0.3 duration at 0.5 Hz = 0.6s
        self.compression_phase_duration = 0.3 / self.freq
        self.extension_phase_duration = 0.3 / self.freq
        
        # Integrated vertical displacement during compression (approximate)
        # Average velocity over sinusoidal profile is (2/pi) * peak
        self.compression_z_displacement = (2.0 / np.pi) * self.compression_vz * self.compression_phase_duration
        
        # Integrated pitch during compression
        self.compression_pitch_displacement = (2.0 / np.pi) * self.compression_pitch_rate * self.compression_phase_duration

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent velocity commands.
        """
        
        if phase < 0.3:
            progress = phase / 0.3
            smooth = 0.5 * (1.0 - np.cos(np.pi * progress))
            vx = self.compression_vx * np.sin(np.pi * progress)
            vz = self.compression_vz * np.sin(np.pi * progress)
            pitch_rate = self.compression_pitch_rate * np.sin(np.pi * progress)
            
        elif phase < 0.5:
            vx = 0.0
            vz = 0.0
            pitch_rate = 0.0
            
        elif phase < 0.8:
            progress = (phase - 0.5) / 0.3
            smooth = 0.5 * (1.0 - np.cos(np.pi * progress))
            vx = self.extension_vx * np.sin(np.pi * progress)
            vz = self.extension_vz * np.sin(np.pi * progress)
            pitch_rate = self.extension_pitch_rate * np.sin(np.pi * progress)
            
        else:
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
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a front or rear leg
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Estimate horizontal distance from base center for pitch compensation
        foot_x_offset = foot[0]  # Initial x position relative to base
        
        if is_front:
            # Front legs: stance during compression, swing during extension
            if phase < 0.5:
                # Stance phase during compression and compressed hold
                progress = phase / 0.5
                x_offset = -self.extension_step_length * 0.5 * progress
                foot[0] += x_offset
                
                # Compensate for base descent during compression
                if phase >= 0.3:
                    # During compressed hold, maintain ground contact despite earlier descent
                    z_compensation = -self.compression_z_displacement * 0.7
                    foot[2] += z_compensation
                else:
                    # During active compression, gradually apply compensation
                    comp_progress = phase / 0.3
                    z_compensation = -self.compression_z_displacement * 0.7 * comp_progress
                    foot[2] += z_compensation
                
            elif phase < 0.8:
                # Swing phase during extension
                swing_progress = (phase - 0.5) / 0.3
                swing_smooth = 0.5 * (1.0 - np.cos(np.pi * swing_progress))
                
                x_start = -self.extension_step_length * 0.5
                x_end = self.extension_step_length * 0.5
                foot[0] += x_start + (x_end - x_start) * swing_smooth
                
                swing_angle = np.pi * swing_progress
                foot[2] += self.swing_height * np.sin(swing_angle)
                
            else:
                # Stance phase during extended hold
                foot[0] += self.extension_step_length * 0.5
                
        else:
            # Rear legs: swing during compression, stance afterward
            if phase < 0.3:
                # Swing phase during compression
                swing_progress = phase / 0.3
                swing_smooth = 0.5 * (1.0 - np.cos(np.pi * swing_progress))
                
                x_start = -self.compression_step_length * 0.5
                x_end = self.compression_step_length * 0.5
                foot[0] += x_start + (x_end - x_start) * swing_smooth
                
                swing_angle = np.pi * swing_progress
                swing_z = self.swing_height * np.sin(swing_angle)
                foot[2] += swing_z
                
                # Pre-compensate landing position for upcoming base descent
                if swing_progress > 0.5:
                    landing_progress = (swing_progress - 0.5) / 0.5
                    z_precomp = -self.compression_z_displacement * 0.8 * landing_progress
                    foot[2] += z_precomp
                
            elif phase < 0.5:
                # Stance during compressed hold
                foot[0] += self.compression_step_length * 0.5
                
                # Compensate for base having descended
                z_compensation = -self.compression_z_displacement * 0.8
                
                # Compensate for pitch-induced lowering of rear feet
                # Rear feet are typically behind base center (negative x in body frame)
                pitch_z_effect = abs(foot_x_offset) * abs(self.compression_pitch_displacement) * 0.5
                z_compensation += pitch_z_effect
                
                foot[2] += z_compensation
                
            elif phase < 0.8:
                # Stance phase during extension
                progress = (phase - 0.5) / 0.3
                smooth_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
                
                x_start = self.compression_step_length * 0.5
                x_end = -self.extension_step_length * 0.3
                foot[0] += x_start + (x_end - x_start) * smooth_progress
                
                # Gradually reduce z compensation as base rises
                z_comp_start = -self.compression_z_displacement * 0.8
                pitch_z_effect = abs(foot_x_offset) * abs(self.compression_pitch_displacement) * 0.5
                z_comp_start += pitch_z_effect
                
                z_comp_end = 0.0
                z_compensation = z_comp_start + (z_comp_end - z_comp_start) * smooth_progress
                foot[2] += z_compensation
                
            else:
                # Stance during extended hold
                foot[0] += -self.extension_step_length * 0.3
                # Minimal compensation needed as base has returned to neutral height
        
        return foot