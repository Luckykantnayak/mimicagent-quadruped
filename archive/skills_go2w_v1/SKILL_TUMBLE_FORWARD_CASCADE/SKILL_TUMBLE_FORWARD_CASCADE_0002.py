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
        self.freq = 0.8
        
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Reduced tuck parameters to stay within joint limits
        self.tuck_offset_x = 0.05
        self.tuck_offset_z = -0.10  # Reduced from -0.15 to avoid extreme joint angles
        self.tuck_radial_scale = 0.55  # Reduced from 0.3 to 0.55 for less aggressive tuck
        
        # Reduced extension parameters to avoid joint limit violations
        self.thrust_extension = 0.06  # Reduced from 0.08
        self.landing_extension_forward = 0.09  # Reduced from 0.12
        self.landing_extension_down = 0.04  # Reduced from 0.05
        
        # Ground clearance safety margin for front legs during launch
        self.launch_lift_margin = 0.05
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Enhanced launch parameters to prevent ground penetration
        self.forward_velocity_max = 1.2
        self.vertical_velocity_launch = 0.55  # Increased from 0.4 for better altitude gain
        self.pitch_rate_max = 4.5
        
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
        
        # Phase 0.0-0.2: Launch with delayed pitch and sustained vertical velocity
        if phase < 0.2:
            progress = phase / 0.2
            # Quadratic pitch rate buildup to delay aggressive forward rotation
            pitch_progress = progress * progress
            vx = self.forward_velocity_max * (0.3 + 0.7 * progress)
            # Sustained vertical velocity with gentler decay to maintain altitude
            vz = self.vertical_velocity_launch * (1.0 - 0.5 * progress)
            pitch_rate = self.pitch_rate_max * pitch_progress
        
        # Phase 0.2-0.5: Active rotation tucked
        elif phase < 0.5:
            progress = (phase - 0.2) / 0.3
            vx = self.forward_velocity_max
            vz = self.vertical_velocity_launch * 0.5 * (1.0 - 2.0 * progress)
            pitch_rate = self.pitch_rate_max
        
        # Phase 0.5-0.7: Inverted transition
        elif phase < 0.7:
            progress = (phase - 0.5) / 0.2
            vx = self.forward_velocity_max * (1.0 - 0.2 * progress)
            vz = -self.vertical_velocity_launch * 0.5 * progress
            pitch_rate = self.pitch_rate_max * (1.0 - 0.3 * progress)
        
        # Phase 0.7-0.9: Landing preparation and contact
        elif phase < 0.9:
            progress = (phase - 0.7) / 0.2
            vx = self.forward_velocity_max * 0.8 * (1.0 - 0.6 * progress)
            vz = -self.vertical_velocity_launch * 0.5 * (1.0 - progress)
            pitch_rate = self.pitch_rate_max * 0.7 * (1.0 - progress)
        
        # Phase 0.9-1.0: Stabilization
        else:
            progress = (phase - 0.9) / 0.1
            vx = self.forward_velocity_max * 0.32 * (1.0 - progress)
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
        
        self.accumulated_pitch += pitch_rate * dt

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase and leg role.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_rear = leg_name.startswith('RL') or leg_name.startswith('RR')
        
        # FRONT LEGS (FL, FR) trajectory
        if is_front:
            # Phase 0.0-0.2: Lift first, then retract inward
            if phase < 0.2:
                progress = phase / 0.2
                # Smooth cubic easing for lift and retraction
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                
                # Initial lift for ground clearance, then gradual retraction
                if progress < 0.4:
                    # First 40% of phase: prioritize lifting
                    lift_progress = progress / 0.4
                    foot[0] = base_pos[0]
                    foot[2] = base_pos[2] + self.launch_lift_margin * lift_progress
                else:
                    # Remaining 60%: retract while maintaining lift
                    retract_progress = (progress - 0.4) / 0.6
                    foot[0] = base_pos[0] * (1.0 - 0.45 * retract_progress) + self.tuck_offset_x * retract_progress
                    foot[2] = base_pos[2] + self.launch_lift_margin + (self.tuck_offset_z - self.launch_lift_margin) * retract_progress
            
            # Phase 0.2-0.7: Tucked during rotation with dynamic depth
            elif phase < 0.7:
                mid_phase = (phase - 0.2) / 0.5
                # Maximum tuck at mid-rotation (around phase 0.45), ease in and out
                tuck_depth = 1.0 - 0.2 * abs(mid_phase - 0.5) / 0.5
                foot[0] = base_pos[0] * self.tuck_radial_scale + self.tuck_offset_x
                foot[2] = base_pos[2] + self.tuck_offset_z * tuck_depth
            
            # Phase 0.7-1.0: Extend toward nominal stance
            else:
                progress = (phase - 0.7) / 0.3
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                tuck_x = base_pos[0] * self.tuck_radial_scale + self.tuck_offset_x
                tuck_z = base_pos[2] + self.tuck_offset_z * 0.8
                foot[0] = tuck_x * (1.0 - smooth_progress) + base_pos[0] * smooth_progress
                foot[2] = tuck_z * (1.0 - smooth_progress) + base_pos[2] * smooth_progress
        
        # REAR LEGS (RL, RR) trajectory
        elif is_rear:
            # Phase 0.0-0.15: Thrust extension primarily downward
            if phase < 0.15:
                progress = phase / 0.15
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                foot[0] = base_pos[0] - self.thrust_extension * 0.3 * smooth_progress
                foot[2] = base_pos[2] - self.thrust_extension * smooth_progress
            
            # Phase 0.15-0.7: Transition to tucked position
            elif phase < 0.7:
                progress = (phase - 0.15) / 0.55
                
                if progress < 0.35:
                    # Smooth transition from thrust to tuck
                    blend = progress / 0.35
                    smooth_blend = blend * blend * (3.0 - 2.0 * blend)
                    thrust_x = base_pos[0] - self.thrust_extension * 0.3
                    thrust_z = base_pos[2] - self.thrust_extension
                    tuck_x = base_pos[0] * self.tuck_radial_scale + self.tuck_offset_x
                    tuck_z = base_pos[2] + self.tuck_offset_z
                    foot[0] = thrust_x * (1.0 - smooth_blend) + tuck_x * smooth_blend
                    foot[2] = thrust_z * (1.0 - smooth_blend) + tuck_z * smooth_blend
                else:
                    # Tucked with dynamic depth
                    tuck_progress = (progress - 0.35) / 0.65
                    tuck_depth = 1.0 - 0.2 * abs(tuck_progress - 0.5) / 0.5
                    foot[0] = base_pos[0] * self.tuck_radial_scale + self.tuck_offset_x
                    foot[2] = base_pos[2] + self.tuck_offset_z * tuck_depth
            
            # Phase 0.7-0.9: Extend for landing (now geometrically front after rotation)
            elif phase < 0.9:
                progress = (phase - 0.7) / 0.2
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                tuck_x = base_pos[0] * self.tuck_radial_scale + self.tuck_offset_x
                tuck_z = base_pos[2] + self.tuck_offset_z * 0.8
                landing_x = base_pos[0] + self.landing_extension_forward
                landing_z = base_pos[2] - self.landing_extension_down
                foot[0] = tuck_x * (1.0 - smooth_progress) + landing_x * smooth_progress
                foot[2] = tuck_z * (1.0 - smooth_progress) + landing_z * smooth_progress
            
            # Phase 0.9-1.0: Stabilize at forward stance
            else:
                progress = (phase - 0.9) / 0.1
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                landing_x = base_pos[0] + self.landing_extension_forward
                landing_z = base_pos[2] - self.landing_extension_down
                final_x = base_pos[0] + 0.06
                final_z = base_pos[2]
                foot[0] = landing_x * (1.0 - smooth_progress) + final_x * smooth_progress
                foot[2] = landing_z * (1.0 - smooth_progress) + final_z * smooth_progress
        
        return foot

    def get_target_positions(self, dt):
        phase = (self.t * self.freq) % 1.0
        self.update_base_motion(phase, dt)
        
        foot_positions_body = {}
        for leg_name in self.leg_names:
            foot_positions_body[leg_name] = self.compute_foot_position_body_frame(leg_name, phase)
        
        self.t += dt
        
        return self.root_pos, self.root_quat, foot_positions_body