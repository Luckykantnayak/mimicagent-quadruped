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
        
        # Launch parameters tuned for rapid liftoff while staying in envelope
        self.launch_vx = 1.2
        self.launch_vz = 1.85  # Increased from 1.6 for faster ground clearance
        
        # Pitch rotation tuned for exactly 2*pi over full cycle
        self.pitch_rate_max = 10.0  # rad/s
        
        # Leg retraction parameters - increased tuck height for better clearance
        self.tuck_height = 0.16  # Increased from 0.12
        self.tuck_longitudinal = 0.05  # Slightly reduced
        
        # Landing compliance
        self.landing_extension = 0.02
        
        # Target landing height
        self.target_landing_height = 0.30
        
    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Height-aware velocity to stay within [0.1, 0.68] m envelope.
        Pitch rotation delayed to allow vertical liftoff first.
        """
        
        vx = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        current_height = self.root_pos[2]
        
        if phase < 0.15:
            # Launch phase: prioritize vertical liftoff, delay pitch rotation
            progress = phase / 0.15
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            vx = self.launch_vx * 0.8
            # Sustain upward velocity longer during launch
            vz = self.launch_vz * (1.0 - 0.5 * smooth)
            # Delay pitch initiation: only start after phase 0.08
            if phase < 0.08:
                pitch_rate = 0.0
            else:
                pitch_progress = (phase - 0.08) / (0.15 - 0.08)
                pitch_rate = self.pitch_rate_max * (0.5 - 0.5 * np.cos(np.pi * pitch_progress))
            
            # Height safety: reduce upward velocity if approaching ceiling
            if current_height > 0.50:
                vz *= max(0.0, (0.68 - current_height) / 0.18)
            
        elif phase < 0.5:
            # Aerial rotation through inverted: sustain pitch rate
            progress = (phase - 0.15) / (0.5 - 0.15)
            vx = self.launch_vx * 0.8
            # Parabolic vertical trajectory
            vz = self.launch_vz * (0.5 - 1.5 * progress)
            pitch_rate = self.pitch_rate_max
            
            # Height safety: clamp velocity near boundaries
            if current_height > 0.60:
                vz = min(vz, -0.5)
            elif current_height < 0.18:
                vz = max(vz, 0.3)
            
        elif phase < 0.70:
            # Complete rotation, begin deceleration of pitch
            progress = (phase - 0.5) / (0.70 - 0.5)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            vx = self.launch_vx * 0.6 * (1.0 - smooth * 0.5)
            vz = -self.launch_vz * 0.35
            pitch_rate = self.pitch_rate_max * (1.0 - smooth * 0.7)
            
            # Height-aware descent
            if current_height < 0.25:
                vz *= 0.4
            
        elif phase < 0.85:
            # Transition to landing: pitch rate goes to zero, gentle descent
            progress = (phase - 0.70) / (0.85 - 0.70)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            vx = self.launch_vx * 0.3 * (1.0 - smooth)
            height_error = current_height - self.target_landing_height
            vz = -0.6 * np.tanh(height_error / 0.1)
            pitch_rate = self.pitch_rate_max * 0.3 * (1.0 - smooth)
            
        else:
            # Final landing phase: zero rotation, stabilize height
            progress = (phase - 0.85) / (1.0 - 0.85)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            vx = 0.2 * (1.0 - smooth)
            height_error = current_height - self.target_landing_height
            vz = -0.4 * np.tanh(height_error / 0.08) * (1.0 - smooth * 0.8)
            pitch_rate = 0.0
            
            # Prevent sinking below minimum during landing
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
        
        # Phase-dependent height clamping
        if phase < 0.15:
            # Relaxed floor during launch to allow natural motion
            min_height = 0.10
        else:
            # Stricter floor after liftoff
            min_height = 0.16
        
        self.root_pos[2] = np.clip(self.root_pos[2], min_height, 0.65)
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot trajectory in body frame throughout flip.
        Leg retraction delayed to phase 0.16 to ensure base has lifted clear.
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine if front or rear leg
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        if phase < 0.16:
            # Extended stance phase: keep legs down longer to ensure ground clearance
            # Nominal position on ground, no retraction yet
            foot = base_pos.copy()
            
        elif phase < 0.32:
            # Launch and retract: legs tuck upward (delayed from 0.1-0.25 to 0.16-0.32)
            progress = (phase - 0.16) / (0.32 - 0.16)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Prioritize upward retraction with increased height
            foot[2] += self.tuck_height * smooth
            
            # Reduced and delayed longitudinal retraction
            longitudinal_factor = smooth * 0.7  # Scale down rearward motion
            if is_front:
                foot[0] -= self.tuck_longitudinal * longitudinal_factor
            else:
                foot[0] += self.tuck_longitudinal * longitudinal_factor
            
        elif phase < 0.5:
            # Airborne through inverted: maintain tucked configuration
            progress = (phase - 0.32) / (0.5 - 0.32)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Maintain full tuck height
            foot[2] += self.tuck_height
            
            # Shift forward in body frame to compensate for rotation
            longitudinal_shift = 0.09 * smooth
            foot[0] += longitudinal_shift
            
        elif phase < 0.70:
            # Completing rotation: begin extending toward landing
            progress = (phase - 0.5) / (0.70 - 0.5)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Reduce tuck height gradually
            tuck_blend = 1.0 - smooth
            foot[2] += self.tuck_height * tuck_blend
            
            # Return toward nominal longitudinal position
            current_offset = 0.09
            if is_front:
                target_offset = -self.tuck_longitudinal * 0.7
            else:
                target_offset = self.tuck_longitudinal * 0.7
            foot[0] = base_pos[0] + current_offset * (1.0 - smooth) + target_offset * (1.0 - smooth)
            
        elif phase < 0.85:
            # Prepare landing: extend legs smoothly
            progress = (phase - 0.70) / (0.85 - 0.70)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Return to nominal x position
            foot[0] = base_pos[0]
            
            # Conservative downward extension
            foot[2] = base_pos[2] - self.landing_extension * smooth * 0.3
            
        else:
            # Final landing: full extension with safety margin
            progress = (phase - 0.85) / (1.0 - 0.85)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Return to nominal stance position
            foot[0] = base_pos[0]
            foot[2] = base_pos[2] - self.landing_extension * smooth
        
        return foot