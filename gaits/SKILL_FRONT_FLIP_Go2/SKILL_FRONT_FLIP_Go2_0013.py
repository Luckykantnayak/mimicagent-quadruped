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
        
        # Launch parameters
        self.launch_vx = 1.5
        self.launch_vz = 3.0
        
        # Pitch rotation parameters (tuned to achieve ~360 degrees over cycle)
        self.pitch_rate_max = 12.0  # rad/s
        
        # Leg retraction parameters
        self.tuck_height = 0.15  # How much legs retract upward
        self.tuck_longitudinal = 0.08  # How much legs move fore/aft during tuck
        
        # Landing compliance
        self.landing_extension = 0.05  # Extra downward extension for landing
        
    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        Phase [0.0, 0.15]: Launch with upward and forward velocity, begin pitch
        Phase [0.15, 0.5]: Sustain pitch rate, ballistic trajectory
        Phase [0.5, 0.75]: Continue pitch rate, descending
        Phase [0.75, 1.0]: Decelerate pitch rate to zero, landing
        """
        
        vx = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        if phase < 0.15:
            # Launch phase: upward and forward impulse, initiate pitch
            progress = phase / 0.15
            vx = self.launch_vx * (1.0 - progress * 0.3)
            vz = self.launch_vz * (1.0 - progress)
            pitch_rate = self.pitch_rate_max * progress
            
        elif phase < 0.5:
            # Aerial rotation through inverted: sustain pitch rate
            # Ballistic trajectory: forward velocity continues, vertical follows parabola
            progress = (phase - 0.15) / (0.5 - 0.15)
            vx = self.launch_vx * 0.7
            # Vertical velocity transitions from upward to downward
            vz = self.launch_vz * (1.0 - 2.0 * progress) * 0.5
            pitch_rate = self.pitch_rate_max
            
        elif phase < 0.75:
            # Complete rotation, descending
            progress = (phase - 0.5) / (0.75 - 0.5)
            vx = self.launch_vx * 0.7 * (1.0 - progress * 0.5)
            vz = -self.launch_vz * 0.6  # Descending
            pitch_rate = self.pitch_rate_max * (1.0 - progress * 0.5)
            
        else:
            # Landing phase: decelerate rotation to zero, dissipate velocities
            progress = (phase - 0.75) / (1.0 - 0.75)
            vx = self.launch_vx * 0.35 * (1.0 - progress)
            vz = -self.launch_vz * 0.6 * (1.0 - progress * 0.8)
            pitch_rate = self.pitch_rate_max * 0.5 * (1.0 - progress)
        
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
        Compute foot trajectory in body frame throughout flip.
        
        All legs follow symmetric motion:
        - [0.0, 0.1]: Stance on ground
        - [0.1, 0.25]: Retract upward and toward body center
        - [0.25, 0.5]: Reposition to compensate for body rotation through inverted
        - [0.5, 0.75]: Extend downward preparing for landing
        - [0.75, 1.0]: Full extension, ground contact, stabilize
        """
        
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine if front or rear leg
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        if phase < 0.1:
            # Stance phase: nominal position on ground
            foot = base_pos.copy()
            
        elif phase < 0.25:
            # Launch and retract: legs tuck upward and toward body
            progress = (phase - 0.1) / (0.25 - 0.1)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Retract upward
            foot[2] += self.tuck_height * smooth
            
            # Move toward body center longitudinally
            if is_front:
                # Front legs move slightly rearward
                foot[0] -= self.tuck_longitudinal * smooth
            else:
                # Rear legs move slightly forward
                foot[0] += self.tuck_longitudinal * smooth
            
        elif phase < 0.5:
            # Airborne through inverted: maintain tucked configuration
            # Compensate for body pitch rotation by adjusting foot position in body frame
            progress = (phase - 0.25) / (0.5 - 0.25)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Maintain tuck height
            foot[2] += self.tuck_height
            
            # As body rotates forward, feet shift forward in body frame to stay reachable
            longitudinal_shift = 0.12 * smooth
            foot[0] += longitudinal_shift
            
        elif phase < 0.75:
            # Completing rotation: transition from tuck to landing extension
            progress = (phase - 0.5) / (0.75 - 0.5)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Reduce tuck height, extend downward
            tuck_blend = 1.0 - smooth
            foot[2] += self.tuck_height * tuck_blend
            
            # Return toward nominal longitudinal position
            if is_front:
                foot[0] = base_pos[0] + (0.12 - self.tuck_longitudinal) * (1.0 - smooth)
            else:
                foot[0] = base_pos[0] + (0.12 + self.tuck_longitudinal) * (1.0 - smooth)
            
        else:
            # Landing phase: extend fully downward, absorb impact
            progress = (phase - 0.75) / (1.0 - 0.75)
            smooth = 0.5 - 0.5 * np.cos(np.pi * progress)
            
            # Extra downward extension for landing compliance
            foot[2] = base_pos[2] - self.landing_extension * (1.0 - smooth)
            
            # Return to nominal longitudinal position
            foot[0] = base_pos[0]
        
        return foot