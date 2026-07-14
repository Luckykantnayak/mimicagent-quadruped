from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_INVERTED_WAVE_HANDSTAND_MotionGenerator(BaseMotionGenerator):
    """
    Inverted wave handstand with lateral carving motion.
    
    Phase structure:
      [0.0–0.2]: Inversion transition - pitch forward into handstand
      [0.2–0.6]: Left carve wave - right rear extends, left rear retracts
      [0.6–1.0]: Right carve wave - left rear extends, right rear retracts
    
    Front legs remain stationary in vertical stance throughout.
    Rear legs alternate lateral sweeps to generate roll/yaw oscillations.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower cycle for dramatic inverted motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.inversion_phase_end = 0.2
        self.left_carve_end = 0.6
        
        # Base motion parameters
        self.forward_vel_inversion = 0.3  # Slight forward drift during inversion
        self.pitch_rate_inversion = 6.5  # Reduced from 8.0 for more controlled rotation
        self.descent_vel_inversion = 0.35  # Downward velocity to maintain ground contact
        
        # Carving motion parameters
        self.lateral_vel_amp = 0.4  # Lateral velocity amplitude
        self.roll_rate_amp = 1.2  # Roll rate amplitude (~15-30 deg oscillation)
        self.yaw_rate_amp = 0.8  # Yaw rate amplitude
        
        # Rear leg lateral sweep parameters
        self.rear_leg_lateral_sweep = 0.15  # Lateral sweep amplitude
        self.rear_leg_fore_aft_sweep = 0.08  # Fore-aft modulation
        
        # Front leg offset (forward projection for inverted support)
        self.front_leg_forward_offset = 0.05
        self.front_leg_extension_inversion = 0.12  # Downward extension during inversion
        
        # Rear leg ground tracking offset
        self.rear_leg_ground_offset = 0.08  # Downward offset to maintain contact
        
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
            # Phase [0.0–0.2]: Inversion transition with vertical compensation
            progress = phase / self.inversion_phase_end
            # Smooth ramp up and down for pitch rate
            pitch_envelope = np.sin(np.pi * progress)
            
            vx = self.forward_vel_inversion * pitch_envelope
            vy = 0.0
            # Coordinated downward velocity to counteract geometric lifting during pitch
            vz = -self.descent_vel_inversion * pitch_envelope
            
            roll_rate = 0.0
            pitch_rate = self.pitch_rate_inversion * pitch_envelope
            yaw_rate = 0.0
            
        elif phase < self.left_carve_end:
            # Phase [0.2–0.6]: Left carve wave
            carve_phase = (phase - self.inversion_phase_end) / (self.left_carve_end - self.inversion_phase_end)
            carve_progress = 2.0 * np.pi * carve_phase
            
            vx = 0.0
            vy = -self.lateral_vel_amp * np.sin(carve_progress)  # Leftward
            vz = 0.0
            
            roll_rate = -self.roll_rate_amp * np.sin(carve_progress)  # Negative roll (left lean)
            pitch_rate = 0.0
            yaw_rate = -self.yaw_rate_amp * np.sin(carve_progress)  # Negative yaw (left turn)
            
        else:
            # Phase [0.6–1.0]: Right carve wave
            carve_phase = (phase - self.left_carve_end) / (1.0 - self.left_carve_end)
            carve_progress = 2.0 * np.pi * carve_phase
            
            vx = 0.0
            vy = self.lateral_vel_amp * np.sin(carve_progress)  # Rightward
            vz = 0.0
            
            roll_rate = self.roll_rate_amp * np.sin(carve_progress)  # Positive roll (right lean)
            pitch_rate = 0.0
            yaw_rate = self.yaw_rate_amp * np.sin(carve_progress)  # Positive yaw (right turn)
        
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
        
        Front legs (FL, FR): Stationary vertical stance with forward offset and vertical extension
        Rear legs (RL, RR): Alternating lateral wave sweeps with ground tracking
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Front legs: stationary with forward offset and vertical extension for inverted support
        if leg_name.startswith('FL') or leg_name.startswith('FR'):
            foot[0] += self.front_leg_forward_offset
            
            # Progressive downward extension during inversion phase
            if phase < self.inversion_phase_end:
                progress = phase / self.inversion_phase_end
                extension_envelope = np.sin(np.pi * 0.5 * progress)  # Smooth ramp to max
                foot[2] -= self.front_leg_extension_inversion * extension_envelope
            else:
                # Maintain full extension during carve phases
                foot[2] -= self.front_leg_extension_inversion
            
            return foot
        
        # Rear legs: alternating lateral waves with ground tracking
        # Apply ground tracking offset throughout all phases
        if phase < self.inversion_phase_end:
            # Phase [0.0–0.2]: Progressive ground tracking during inversion
            progress = phase / self.inversion_phase_end
            ground_offset_envelope = np.sin(np.pi * 0.5 * progress)
            foot[2] -= self.rear_leg_ground_offset * ground_offset_envelope
            return foot
        
        # Full ground offset during carve phases
        foot[2] -= self.rear_leg_ground_offset
        
        if phase < self.left_carve_end:
            # Phase [0.2–0.6]: Left carve - RR extends out, RL retracts in
            carve_phase = (phase - self.inversion_phase_end) / (self.left_carve_end - self.inversion_phase_end)
            # Use smooth sinusoidal progression for better continuity
            wave_progress = 0.5 * (1.0 - np.cos(np.pi * carve_phase))
            
            if leg_name.startswith('RL'):
                # Left rear: retract inward (positive Y) and slightly forward
                lateral_offset = self.rear_leg_lateral_sweep * wave_progress
                fore_aft_offset = self.rear_leg_fore_aft_sweep * wave_progress
                foot[1] += lateral_offset  # Inward (toward midline)
                foot[0] += fore_aft_offset  # Slightly forward
                
            elif leg_name.startswith('RR'):
                # Right rear: extend outward (negative Y) and slightly rearward
                lateral_offset = -self.rear_leg_lateral_sweep * wave_progress
                fore_aft_offset = -self.rear_leg_fore_aft_sweep * wave_progress
                foot[1] += lateral_offset  # Outward (away from midline)
                foot[0] += fore_aft_offset  # Slightly rearward
        
        else:
            # Phase [0.6–1.0]: Right carve - RL extends out, RR retracts in
            carve_phase = (phase - self.left_carve_end) / (1.0 - self.left_carve_end)
            wave_progress = 0.5 * (1.0 - np.cos(np.pi * carve_phase))
            
            if leg_name.startswith('RL'):
                # Left rear: extend outward (negative Y) and slightly rearward
                lateral_offset = -self.rear_leg_lateral_sweep * wave_progress
                fore_aft_offset = -self.rear_leg_fore_aft_sweep * wave_progress
                foot[1] += lateral_offset  # Outward
                foot[0] += fore_aft_offset  # Slightly rearward
                
            elif leg_name.startswith('RR'):
                # Right rear: retract inward (positive Y) and slightly forward
                lateral_offset = self.rear_leg_lateral_sweep * wave_progress
                fore_aft_offset = self.rear_leg_fore_aft_sweep * wave_progress
                foot[1] += lateral_offset  # Inward
                foot[0] += fore_aft_offset  # Slightly forward
        
        return foot