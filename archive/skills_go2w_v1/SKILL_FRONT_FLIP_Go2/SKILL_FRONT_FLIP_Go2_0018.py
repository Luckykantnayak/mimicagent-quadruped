from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_FRONT_FLIP_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Front flip kinematic motion generator.
    
    Phases:
      [0.0, 0.25]: Launch and initiate rotation
      [0.25, 0.5]: Forward rotation through inverted
      [0.5, 0.75]: Complete rotation and prepare landing
      [0.75, 1.0]: Landing and stabilization
    
    - Base undergoes 360-degree forward pitch rotation
    - All legs airborne during phases [0.1, 0.85]
    - Legs reposition in body frame to maintain kinematic feasibility
    - Landing re-establishes contact and stabilizes pose
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for aerial maneuver
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Launch parameters - micro-tuned for optimal clearance
        self.launch_vx = 1.2
        self.launch_vz = 1.75  # Increased from 1.7 for faster clearance
        
        # Pitch rotation tuned for exactly 2*pi over full cycle
        self.pitch_rate_max = 10.0  # rad/s
        
        # Leg retraction parameters - micro-tuned balance
        self.tuck_height = 0.145  # Increased from 0.14, still safe from joint limits
        self.tuck_longitudinal = 0.06
        
        # Pitch compensation - micro-increased within safe margin
        self.pitch_compensation_max = 0.11  # Increased from 0.10
        
        # Landing compliance
        self.landing_extension = 0.02
        
        # Target landing height
        self.target_landing_height = 0.30
        
    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Height-aware velocity to stay within [0.1, 0.68] m envelope.
        """
        
        vx = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        current_height = self.root_pos[2]
        
        if phase < 0.15:
            # Launch phase: upward and forward impulse, synchronized pitch
            progress = phase / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            vx = self.launch_vx
            vz = self.launch_vz * (1.0 - smooth * 0.6)
            pitch_rate = self.pitch_rate_max * smooth
            
            # Height safety: reduce upward velocity if approaching ceiling
            if current_height > 0.50:
                vz *= max(0.0, (0.68 - current_height) / 0.18)
            
        elif phase < 0.5:
            # Aerial rotation through inverted: sustain pitch rate
            progress = (phase - 0.15) / (0.5 - 0.15)
            vx = self.launch_vx * 0.8
            vz = self.launch_vz * (0.55 - 1.6 * progress)
            pitch_rate = self.pitch_rate_max
            
            # Height safety
            if current_height > 0.60:
                vz = min(vz, -0.5)
            elif current_height < 0.18:
                vz = max(vz, 0.4)
            
        elif phase < 0.70:
            # Complete rotation, decelerate pitch
            progress = (phase - 0.5) / (0.70 - 0.5)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            vx = self.launch_vx * 0.6 * (1.0 - smooth * 0.5)
            vz = -self.launch_vz * 0.35
            pitch_rate = self.pitch_rate_max * (1.0 - smooth * 0.7)
            
            if current_height < 0.25:
                vz *= 0.4
            
        elif phase < 0.85:
            # Transition to landing: pitch rate to zero
            progress = (phase - 0.70) / (0.85 - 0.70)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            vx = self.launch_vx * 0.3 * (1.0 - smooth)
            height_error = current_height - self.target_landing_height
            vz = -0.6 * np.tanh(height_error / 0.1)
            pitch_rate = self.pitch_rate_max * 0.3 * (1.0 - smooth)
            
        else:
            # Final landing: stabilize
            progress = (phase - 0.85) / (1.0 - 0.85)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            vx = 0.2 * (1.0 - smooth)
            height_error = current_height - self.target_landing_height
            vz = -0.4 * np.tanh(height_error / 0.08) * (1.0 - smooth * 0.8)
            pitch_rate = 0.0
            
            if current_height < 0.16:
                vz = max(vz, 0.0)
        
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Post-integration height clamping with micro-raised floor
        self.root_pos[2] = np.clip(self.root_pos[2], 0.15, 0.65)
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot trajectory in body frame throughout flip.
        Micro-tuned pitch compensation to minimize penetration while respecting joint limits.
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine if front or rear leg
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        if phase < 0.1:
            # Stance phase: nominal position
            foot = base_pos.copy()
            
        elif phase < 0.25:
            # Launch and retract: legs tuck with moderated pitch compensation
            progress = (phase - 0.1) / (0.25 - 0.1)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Base upward retraction with micro-increased height
            foot[2] += self.tuck_height * smooth
            
            # Micro-tuned pitch compensation for front legs
            if is_front:
                # Estimate pitch angle during this phase
                if phase < 0.15:
                    phase_progress = phase / 0.15
                    estimated_pitch = self.pitch_rate_max * 0.5 * phase_progress * phase / self.freq
                else:
                    pitch_at_015 = self.pitch_rate_max * 0.5 * 0.15 / self.freq
                    additional_pitch = self.pitch_rate_max * (phase - 0.15) / self.freq
                    estimated_pitch = pitch_at_015 + additional_pitch
                
                # Apply micro-increased compensation scaled to pitch angle
                pitch_factor = min(estimated_pitch / (np.pi / 4), 1.0)
                pitch_compensation = self.pitch_compensation_max * pitch_factor
                foot[2] += pitch_compensation * smooth
            
            # Longitudinal retraction
            if is_front:
                foot[0] -= self.tuck_longitudinal * smooth
            else:
                foot[0] += self.tuck_longitudinal * smooth
            
        elif phase < 0.37:
            # Early airborne: maintain compensation slightly longer
            progress = (phase - 0.25) / (0.37 - 0.25)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Maintain full tuck height
            foot[2] += self.tuck_height
            
            # Maintain full pitch compensation for front legs
            if is_front:
                foot[2] += self.pitch_compensation_max
            
            # Begin forward shift for rotation
            longitudinal_shift = 0.08 * smooth
            foot[0] += longitudinal_shift
            
        elif phase < 0.5:
            # Mid airborne: taper compensation as body rotates past vertical
            progress = (phase - 0.37) / (0.5 - 0.37)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Maintain tuck height
            foot[2] += self.tuck_height
            
            # Taper pitch compensation to zero
            if is_front:
                compensation_blend = 1.0 - smooth
                foot[2] += self.pitch_compensation_max * compensation_blend
            
            # Continue forward shift
            longitudinal_shift = 0.08 + 0.04 * smooth
            foot[0] += longitudinal_shift
            
        elif phase < 0.70:
            # Completing rotation: extend toward landing
            progress = (phase - 0.5) / (0.70 - 0.5)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Reduce tuck height gradually
            tuck_blend = 1.0 - smooth
            foot[2] += self.tuck_height * tuck_blend
            
            # Return toward nominal longitudinal position
            current_offset = 0.12
            if is_front:
                target_offset = -self.tuck_longitudinal
            else:
                target_offset = self.tuck_longitudinal
            foot[0] = base_pos[0] + current_offset * (1.0 - smooth) + target_offset * (1.0 - smooth)
            
        elif phase < 0.85:
            # Prepare landing: extend legs
            progress = (phase - 0.70) / (0.85 - 0.70)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Return to nominal position
            foot[0] = base_pos[0]
            foot[2] = base_pos[2] - self.landing_extension * smooth * 0.3
            
        else:
            # Final landing: full extension
            progress = (phase - 0.85) / (1.0 - 0.85)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            foot[0] = base_pos[0]
            foot[2] = base_pos[2] - self.landing_extension * smooth
        
        return foot