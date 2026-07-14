from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_INVERTED_WAVE_HANDSTAND_MotionGenerator(BaseMotionGenerator):
    """
    Inverted wave handstand with lateral carving motion.
    
    Phase structure:
      [0.0–0.3]: Inversion transition - pitch forward into handstand
      [0.3–0.65]: Left carve wave - right rear extends, left rear retracts
      [0.65–1.0]: Right carve wave - left rear extends, right rear retracts
    
    Front legs remain stationary in vertical stance throughout.
    Rear legs alternate lateral sweeps to generate roll/yaw oscillations.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - extended inversion phase for smoother transition
        self.inversion_phase_end = 0.3
        self.left_carve_end = 0.65
        
        # Base motion parameters - adjusted for ground contact maintenance
        self.forward_vel_inversion = 0.25
        self.pitch_rate_inversion = 4.5  # Reduced from 6.5 to slow rotation
        self.descent_vel_inversion = 0.95  # Increased from 0.35 to compensate for geometric lifting
        
        # Carving motion parameters
        self.lateral_vel_amp = 0.35
        self.roll_rate_amp = 1.0
        self.yaw_rate_amp = 0.7
        
        # Rear leg lateral sweep parameters - reduced to avoid joint limits
        self.rear_leg_lateral_sweep = 0.09  # Reduced from 0.15
        self.rear_leg_fore_aft_sweep = 0.04  # Reduced from 0.08
        
        # Front leg offset - reduced for joint limit compliance
        self.front_leg_forward_offset = 0.04
        self.front_leg_extension_inversion = 0.08  # Reduced from 0.12
        
        # Rear leg ground tracking offset - reduced to avoid joint limits
        self.rear_leg_ground_offset = 0.05  # Reduced from 0.08
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity accumulators
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent velocity commands.
        """
        if phase < self.inversion_phase_end:
            # Phase [0.0–0.3]: Inversion transition with aggressive vertical compensation
            progress = phase / self.inversion_phase_end
            # Smooth ramp up and down
            pitch_envelope = np.sin(np.pi * progress)
            
            vx = self.forward_vel_inversion * pitch_envelope
            vy = 0.0
            # Strong downward velocity to maintain ground contact during pitch
            vz = -self.descent_vel_inversion * pitch_envelope
            
            roll_rate = 0.0
            pitch_rate = self.pitch_rate_inversion * pitch_envelope
            yaw_rate = 0.0
            
        elif phase < self.left_carve_end:
            # Phase [0.3–0.65]: Left carve wave
            carve_phase = (phase - self.inversion_phase_end) / (self.left_carve_end - self.inversion_phase_end)
            carve_progress = 2.0 * np.pi * carve_phase
            
            vx = 0.0
            vy = -self.lateral_vel_amp * np.sin(carve_progress)
            vz = 0.0
            
            roll_rate = -self.roll_rate_amp * np.sin(carve_progress)
            pitch_rate = 0.0
            yaw_rate = -self.yaw_rate_amp * np.sin(carve_progress)
            
        else:
            # Phase [0.65–1.0]: Right carve wave
            carve_phase = (phase - self.left_carve_end) / (1.0 - self.left_carve_end)
            carve_progress = 2.0 * np.pi * carve_phase
            
            vx = 0.0
            vy = self.lateral_vel_amp * np.sin(carve_progress)
            vz = 0.0
            
            roll_rate = self.roll_rate_amp * np.sin(carve_progress)
            pitch_rate = 0.0
            yaw_rate = self.yaw_rate_amp * np.sin(carve_progress)
        
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
        Compute foot position in body frame based on phase.
        
        Front legs (FL, FR): Stationary vertical stance with pitch-gated extension
        Rear legs (RL, RR): Alternating lateral wave sweeps with ground tracking
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Front legs: stationary with pitch-gated extension to avoid early ground penetration
        if leg_name.startswith('FL') or leg_name.startswith('FR'):
            foot[0] += self.front_leg_forward_offset
            
            # Extension only activates after pitch has accumulated significantly
            # Gate extension to start ramping after pitch > ~60 degrees (phase > 0.15)
            if phase < self.inversion_phase_end:
                # Map phase to extension: start at phase 0.1, full at phase 0.3
                if phase < 0.1:
                    extension_factor = 0.0
                else:
                    extension_progress = (phase - 0.1) / (self.inversion_phase_end - 0.1)
                    extension_factor = 0.5 * (1.0 - np.cos(np.pi * extension_progress))
                foot[2] -= self.front_leg_extension_inversion * extension_factor
            else:
                # Maintain full extension during carve phases
                foot[2] -= self.front_leg_extension_inversion
            
            return foot
        
        # Rear legs: alternating lateral waves with gradual ground tracking
        # Apply ground tracking with smooth ramp during inversion
        if phase < self.inversion_phase_end:
            # Progressive ground tracking during inversion
            progress = phase / self.inversion_phase_end
            ground_offset_envelope = 0.5 * (1.0 - np.cos(np.pi * progress))
            foot[2] -= self.rear_leg_ground_offset * ground_offset_envelope
            return foot
        
        # Full ground offset during carve phases
        foot[2] -= self.rear_leg_ground_offset
        
        if phase < self.left_carve_end:
            # Phase [0.3–0.65]: Left carve - RR extends out, RL retracts in
            carve_phase = (phase - self.inversion_phase_end) / (self.left_carve_end - self.inversion_phase_end)
            # Smooth sinusoidal progression
            wave_progress = 0.5 * (1.0 - np.cos(np.pi * carve_phase))
            
            if leg_name.startswith('RL'):
                # Left rear: retract inward (positive Y) and slightly forward
                lateral_offset = self.rear_leg_lateral_sweep * wave_progress
                fore_aft_offset = self.rear_leg_fore_aft_sweep * wave_progress
                foot[1] += lateral_offset
                foot[0] += fore_aft_offset
                
            elif leg_name.startswith('RR'):
                # Right rear: extend outward (negative Y) and slightly rearward
                lateral_offset = -self.rear_leg_lateral_sweep * wave_progress
                fore_aft_offset = -self.rear_leg_fore_aft_sweep * wave_progress
                foot[1] += lateral_offset
                foot[0] += fore_aft_offset
        
        else:
            # Phase [0.65–1.0]: Right carve - RL extends out, RR retracts in
            carve_phase = (phase - self.left_carve_end) / (1.0 - self.left_carve_end)
            wave_progress = 0.5 * (1.0 - np.cos(np.pi * carve_phase))
            
            if leg_name.startswith('RL'):
                # Left rear: extend outward (negative Y) and slightly rearward
                lateral_offset = -self.rear_leg_lateral_sweep * wave_progress
                fore_aft_offset = -self.rear_leg_fore_aft_sweep * wave_progress
                foot[1] += lateral_offset
                foot[0] += fore_aft_offset
                
            elif leg_name.startswith('RR'):
                # Right rear: retract inward (positive Y) and slightly forward
                lateral_offset = self.rear_leg_lateral_sweep * wave_progress
                fore_aft_offset = self.rear_leg_fore_aft_sweep * wave_progress
                foot[1] += lateral_offset
                foot[0] += fore_aft_offset
        
        return foot