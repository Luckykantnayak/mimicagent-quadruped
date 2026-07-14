from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_POGO_BOUNCE_FORWARD_MotionGenerator(BaseMotionGenerator):
    """
    Pogo-style bounce with all four legs synchronized.
    
    - All legs move together (zero phase offset)
    - Compression -> Launch -> Flight -> Descent -> Landing absorption
    - Base maintains constant forward velocity with oscillating vertical velocity
    - Feet expressed in BODY frame
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0  # 1 Hz hop frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # All legs synchronized - zero phase offset
        self.phase_offsets = {
            leg_names[0]: 0.0,
            leg_names[1]: 0.0,
            leg_names[2]: 0.0,
            leg_names[3]: 0.0,
        }
        
        # Motion parameters
        self.forward_step = 0.12  # Forward displacement per hop cycle
        self.tuck_height = 0.06  # How much legs tuck during flight
        
        # Vertical velocity parameters - balanced for neutral height maintenance
        self.vz_launch_max = 0.28  # Peak upward velocity during launch
        self.vz_descent_max = -0.28  # Peak downward velocity during descent
        
        # Forward velocity (constant)
        self.vx_forward = 0.5  # Constant forward speed
        
        # Nominal base height target
        self.nominal_base_height = 0.30
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.nominal_base_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Track base height for foot coordination
        self.current_base_height = self.nominal_base_height

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        Vertical velocity oscillates through hop cycle with height stabilization
        to maintain base around nominal height.
        """
        
        vx = self.vx_forward
        vy = 0.0
        vz = 0.0
        
        # Compute vertical velocity based on phase with smooth transitions
        if phase < 0.2:
            # Compression: downward velocity decreasing to zero
            progress = phase / 0.2
            blend = 1.0 - smoothstep(progress)
            vz = self.vz_descent_max * blend
            
        elif phase < 0.4:
            # Launch: upward velocity rapidly increasing
            progress = (phase - 0.2) / 0.2
            blend = smoothstep(progress)
            vz = self.vz_launch_max * blend
            
        elif phase < 0.6:
            # Flight peak: upward through zero to downward
            progress = (phase - 0.4) / 0.2
            # Smooth sinusoidal transition from positive to negative
            vz = self.vz_launch_max * np.cos(np.pi * progress)
            
        elif phase < 0.8:
            # Descent: downward velocity increasing in magnitude
            progress = (phase - 0.6) / 0.2
            blend = smoothstep(progress)
            vz = -self.vz_launch_max * blend
            
        else:
            # Landing absorption: downward velocity decreasing to zero
            progress = (phase - 0.8) / 0.2
            blend = 1.0 - smoothstep(progress)
            vz = self.vz_descent_max * blend
        
        # Height stabilization: add corrective bias to prevent drift
        height_error = self.root_pos[2] - self.nominal_base_height
        height_correction = -0.3 * height_error  # Soft restoring force
        vz += height_correction
        
        # Set velocity commands (world frame)
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Track current base height for foot coordination
        self.current_base_height = self.root_pos[2]

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        
        Contact phases: feet maintain world z=0 by computing body-frame position
        from world constraint and current base height.
        Flight phase: feet move relative to body for aerial maneuver.
        """
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        base_foot_z = foot[2]  # Nominal body-frame z (negative, pointing down)
        
        # Ground clearance target (world frame)
        target_world_z = 0.0
        
        if phase < 0.2:
            # Compression: base lowers, foot stays on ground (world z=0)
            progress = phase / 0.2
            compression_curve = smoothstep(progress)
            
            # Compute body-frame z to maintain world z=0 as base height changes
            # When base descends, body-frame z must become LESS negative (leg shortens)
            foot[2] = target_world_z - self.current_base_height
            
            # Foot moves slightly rearward during compression
            foot[0] -= 0.3 * self.forward_step * compression_curve
            
        elif phase < 0.4:
            # Launch: explosive extension, foot pushes off ground then breaks contact
            progress = (phase - 0.4 + 0.2) / 0.2
            
            # Transition window for liftoff at end of launch phase
            liftoff_transition_start = 0.7
            
            if progress < liftoff_transition_start:
                # Still in contact, extending
                foot[2] = target_world_z - self.current_base_height
                # Return toward neutral stance
                remaining_compression = 1.0 - progress / liftoff_transition_start
                foot[0] -= 0.3 * self.forward_step * remaining_compression
            else:
                # Breaking contact, beginning to lift - smooth transition to aerial mode
                liftoff_progress = (progress - liftoff_transition_start) / (1.0 - liftoff_transition_start)
                liftoff_blend = smoothstep(liftoff_progress)
                
                # Blend from world-constrained to body-relative
                world_constrained_z = target_world_z - self.current_base_height
                body_relative_z = base_foot_z - 0.02 * liftoff_blend
                foot[2] = world_constrained_z * (1.0 - liftoff_blend) + body_relative_z * liftoff_blend
                
                foot[0] += 0.1 * self.forward_step * liftoff_progress
            
        elif phase < 0.6:
            # Flight tuck: foot retracts upward and forward (aerial phase, body-relative)
            progress = (phase - 0.4) / 0.2
            # Smooth tucking motion
            tuck_curve = np.sin(np.pi * progress)
            foot[2] = base_foot_z + self.tuck_height * tuck_curve
            foot[0] += 0.4 * self.forward_step * progress
            
        elif phase < 0.8:
            # Descent extension: foot extends downward and forward for landing (body-relative)
            progress = (phase - 0.6) / 0.2
            # Untuck from peak
            tuck_curve = np.sin(np.pi * (1.0 - progress))
            foot[2] = base_foot_z + self.tuck_height * tuck_curve
            # Reach forward to landing position
            forward_reach = 0.4 + 0.3 * progress
            foot[0] += self.forward_step * forward_reach
            
        else:
            # Landing absorption: transition from aerial to ground contact
            progress = (phase - 0.8) / 0.2
            
            # Transition window for touchdown at start of landing phase
            touchdown_transition_end = 0.3
            
            if progress < touchdown_transition_end:
                # Transitioning to contact - blend from body-relative to world-constrained
                touchdown_blend = smoothstep(progress / touchdown_transition_end)
                
                body_relative_z = base_foot_z
                world_constrained_z = target_world_z - self.current_base_height
                foot[2] = body_relative_z * (1.0 - touchdown_blend) + world_constrained_z * touchdown_blend
            else:
                # Fully in contact, maintaining world z=0 as base compresses
                foot[2] = target_world_z - self.current_base_height
            
            # Foot at forward landing position
            foot[0] += 0.5 * self.forward_step
        
        return foot