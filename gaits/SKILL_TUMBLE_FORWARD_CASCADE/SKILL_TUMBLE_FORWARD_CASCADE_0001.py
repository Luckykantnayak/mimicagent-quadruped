from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_TUMBLE_FORWARD_CASCADE_MotionGenerator(BaseMotionGenerator):
    """
    Forward tumbling motion with 180-degree pitch rotation and leg role swap.
    
    Phase structure:
    - [0.0, 0.2]: Launch and pitch initiation
    - [0.2, 0.5]: Active rotation with legs tucked
    - [0.5, 0.7]: Inverted transition
    - [0.7, 0.9]: Landing preparation and contact
    - [0.9, 1.0]: Stabilization and reset
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency for tumble motion
        
        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Tuck parameters - legs fold close to body center during rotation
        self.tuck_offset_x = 0.05  # Small x offset when tucked
        self.tuck_offset_z = -0.15  # Height when tucked (up from nominal)
        
        # Thrust and landing extension parameters
        self.thrust_extension = 0.08  # Downward extension during thrust
        self.landing_extension_forward = 0.12  # Forward reach for landing legs
        self.landing_extension_down = 0.05  # Downward reach for landing
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Motion parameters for base velocity control
        self.forward_velocity_max = 1.2  # Peak forward velocity during launch
        self.vertical_velocity_launch = 0.4  # Initial upward velocity component
        self.pitch_rate_max = 4.5  # Peak pitch angular velocity (rad/s)
        
        # Track accumulated pitch for coordination
        self.accumulated_pitch = 0.0

    def reset(self, root_pos, root_quat):
        self.root_pos = root_pos.copy()
        self.root_quat = root_quat.copy()
        self.t = 0.0
        self.accumulated_pitch = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on tumble phase.
        """
        vx = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        # Phase 0.0-0.2: Launch and pitch initiation
        if phase < 0.2:
            progress = phase / 0.2
            vx = self.forward_velocity_max * (0.3 + 0.7 * progress)
            vz = self.vertical_velocity_launch * (1.0 - progress)
            pitch_rate = self.pitch_rate_max * progress
        
        # Phase 0.2-0.5: Active rotation tucked
        elif phase < 0.5:
            progress = (phase - 0.2) / 0.3
            vx = self.forward_velocity_max
            vz = self.vertical_velocity_launch * (1.0 - 2.0 * progress)  # Ballistic arc
            pitch_rate = self.pitch_rate_max
        
        # Phase 0.5-0.7: Inverted transition
        elif phase < 0.7:
            progress = (phase - 0.5) / 0.2
            vx = self.forward_velocity_max * (1.0 - 0.2 * progress)
            vz = -self.vertical_velocity_launch * progress  # Descending
            pitch_rate = self.pitch_rate_max * (1.0 - 0.3 * progress)
        
        # Phase 0.7-0.9: Landing preparation and contact
        elif phase < 0.9:
            progress = (phase - 0.7) / 0.2
            vx = self.forward_velocity_max * 0.8 * (1.0 - 0.6 * progress)
            vz = -self.vertical_velocity_launch * (1.0 - progress)  # Arrest vertical motion
            pitch_rate = self.pitch_rate_max * 0.7 * (1.0 - progress)  # Decelerate rotation
        
        # Phase 0.9-1.0: Stabilization
        else:
            progress = (phase - 0.9) / 0.1
            vx = self.forward_velocity_max * 0.32 * (1.0 - progress)
            vz = 0.0
            pitch_rate = 0.0
        
        # Set velocity commands in world frame
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
        
        # Track accumulated pitch rotation for coordination
        self.accumulated_pitch += pitch_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase and leg role.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Identify leg group
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_rear = leg_name.startswith('RL') or leg_name.startswith('RR')
        
        # FRONT LEGS (FL, FR) trajectory
        if is_front:
            # Phase 0.0-0.2: Retract inward and upward
            if phase < 0.2:
                progress = phase / 0.2
                foot[0] = base_pos[0] * (1.0 - 0.7 * progress) + self.tuck_offset_x * progress
                foot[2] = base_pos[2] + self.tuck_offset_z * progress
            
            # Phase 0.2-0.7: Remain tucked during rotation
            elif phase < 0.7:
                foot[0] = base_pos[0] * 0.3 + self.tuck_offset_x
                foot[2] = base_pos[2] + self.tuck_offset_z
            
            # Phase 0.7-1.0: Gradually extend toward nominal stance
            else:
                progress = (phase - 0.7) / 0.3
                foot[0] = (base_pos[0] * 0.3 + self.tuck_offset_x) * (1.0 - progress) + base_pos[0] * progress
                foot[2] = (base_pos[2] + self.tuck_offset_z) * (1.0 - progress) + base_pos[2] * progress
        
        # REAR LEGS (RL, RR) trajectory
        elif is_rear:
            # Phase 0.0-0.15: Thrust extension downward/backward
            if phase < 0.15:
                progress = phase / 0.15
                foot[0] = base_pos[0] - self.thrust_extension * 0.5 * progress
                foot[2] = base_pos[2] - self.thrust_extension * progress
            
            # Phase 0.15-0.7: Retract to tucked position during rotation
            elif phase < 0.7:
                progress = (phase - 0.15) / 0.55
                # Transition from thrust to tuck
                if progress < 0.3:
                    blend = progress / 0.3
                    thrust_x = base_pos[0] - self.thrust_extension * 0.5
                    thrust_z = base_pos[2] - self.thrust_extension
                    tuck_x = base_pos[0] * 0.3 + self.tuck_offset_x
                    tuck_z = base_pos[2] + self.tuck_offset_z
                    foot[0] = thrust_x * (1.0 - blend) + tuck_x * blend
                    foot[2] = thrust_z * (1.0 - blend) + tuck_z * blend
                else:
                    foot[0] = base_pos[0] * 0.3 + self.tuck_offset_x
                    foot[2] = base_pos[2] + self.tuck_offset_z
            
            # Phase 0.7-0.9: Extend forward and downward for landing (now geometrically front)
            elif phase < 0.9:
                progress = (phase - 0.7) / 0.2
                tuck_x = base_pos[0] * 0.3 + self.tuck_offset_x
                tuck_z = base_pos[2] + self.tuck_offset_z
                # After 180° rotation, rear legs are now forward in body frame
                # Extend in positive x (body forward) and negative z (body down)
                landing_x = base_pos[0] + self.landing_extension_forward
                landing_z = base_pos[2] - self.landing_extension_down
                foot[0] = tuck_x * (1.0 - progress) + landing_x * progress
                foot[2] = tuck_z * (1.0 - progress) + landing_z * progress
            
            # Phase 0.9-1.0: Stabilize at nominal stance (now acting as front legs)
            else:
                progress = (phase - 0.9) / 0.1
                landing_x = base_pos[0] + self.landing_extension_forward
                landing_z = base_pos[2] - self.landing_extension_down
                # Settle to a forward stance position
                final_x = base_pos[0] + 0.08
                final_z = base_pos[2]
                foot[0] = landing_x * (1.0 - progress) + final_x * progress
                foot[2] = landing_z * (1.0 - progress) + final_z * progress
        
        return foot