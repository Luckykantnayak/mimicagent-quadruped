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
        
        self.base_feet_pos_body = {}
        for k, v in initial_foot_positions_body.items():
            pos = v.copy()
            is_front = k.startswith('FL') or k.startswith('FR')
            if is_front and pos[2] > -0.04:
                pos[2] = -0.04
            self.base_feet_pos_body[k] = pos
        
        # Relaxed tuck parameters for joint limit compliance
        self.tuck_offset_x = 0.015
        self.tuck_offset_z = -0.045
        self.tuck_radial_scale = 0.78
        
        # Conservative extension parameters
        self.thrust_extension = 0.05
        self.landing_extension_forward = 0.08
        self.landing_extension_down = 0.03
        
        # Ground clearance safety margin
        self.launch_lift_margin = 0.045
        
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Launch parameters
        self.forward_velocity_max = 1.2
        self.vertical_velocity_launch = 0.55
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
            pitch_progress = progress * progress
            vx = self.forward_velocity_max * (0.3 + 0.7 * progress)
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

    def compute_pitch_relief_factor(self, phase):
        """
        Compute relief factor based on phase to reduce tuck depth during high-stress rotation.
        Returns value between 0.35 and 1.0, with minimum at mid-rotation (phase 0.45).
        """
        if phase < 0.2:
            return 1.0
        elif phase < 0.7:
            rotation_phase = (phase - 0.2) / 0.5
            # Maximum relief at mid-rotation where pitch is ~90 degrees
            relief = 1.0 - 0.65 * (1.0 - abs(2.0 * rotation_phase - 1.0))
            return relief
        else:
            return 1.0

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on phase and leg role.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_rear = leg_name.startswith('RL') or leg_name.startswith('RR')
        
        pitch_relief = self.compute_pitch_relief_factor(phase)
        
        # FRONT LEGS (FL, FR) trajectory
        if is_front:
            # Phase 0.0-0.2: Lift first, then retract inward with extended transition
            if phase < 0.2:
                progress = phase / 0.2
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                
                if progress < 0.45:
                    # First 45%: prioritize lifting
                    lift_progress = progress / 0.45
                    lift_smooth = lift_progress * lift_progress * (3.0 - 2.0 * lift_progress)
                    foot[0] = base_pos[0]
                    foot[2] = base_pos[2] - self.launch_lift_margin * lift_smooth
                else:
                    # Remaining 55%: gradual retract while maintaining lift
                    retract_progress = (progress - 0.45) / 0.55
                    retract_smooth = retract_progress * retract_progress * (3.0 - 2.0 * retract_progress)
                    foot[0] = base_pos[0] * (1.0 - 0.22 * retract_smooth) + self.tuck_offset_x * retract_smooth
                    foot[2] = base_pos[2] - self.launch_lift_margin + (self.tuck_offset_z + self.launch_lift_margin) * retract_smooth
            
            # Phase 0.2-0.7: Tucked during rotation with pitch-aware depth relief
            elif phase < 0.7:
                radial_blend = self.tuck_radial_scale + (1.0 - self.tuck_radial_scale) * (1.0 - pitch_relief) * 0.3
                foot[0] = base_pos[0] * radial_blend + self.tuck_offset_x * pitch_relief
                foot[2] = base_pos[2] + self.tuck_offset_z * pitch_relief
            
            # Phase 0.7-1.0: Extend toward nominal stance with gradual transition
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
                foot[0] = base_pos[0] - self.thrust_extension * 0.25 * smooth_progress
                foot[2] = base_pos[2] - self.thrust_extension * smooth_progress
            
            # Phase 0.15-0.7: Extended transition to tucked position
            elif phase < 0.7:
                progress = (phase - 0.15) / 0.55
                
                if progress < 0.4:
                    # Extended smooth transition from thrust to tuck
                    blend = progress / 0.4
                    smooth_blend = blend * blend * (3.0 - 2.0 * blend)
                    thrust_x = base_pos[0] - self.thrust_extension * 0.25
                    thrust_z = base_pos[2] - self.thrust_extension
                    tuck_x = base_pos[0] * self.tuck_radial_scale + self.tuck_offset_x
                    tuck_z = base_pos[2] + self.tuck_offset_z
                    foot[0] = thrust_x * (1.0 - smooth_blend) + tuck_x * smooth_blend
                    foot[2] = thrust_z * (1.0 - smooth_blend) + tuck_z * smooth_blend
                else:
                    # Tucked with pitch-aware depth relief
                    tuck_progress = (progress - 0.4) / 0.6
                    current_phase = 0.15 + progress * 0.55
                    local_relief = self.compute_pitch_relief_factor(current_phase)
                    radial_blend = self.tuck_radial_scale + (1.0 - self.tuck_radial_scale) * (1.0 - local_relief) * 0.3
                    foot[0] = base_pos[0] * radial_blend + self.tuck_offset_x * local_relief
                    foot[2] = base_pos[2] + self.tuck_offset_z * local_relief
            
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
                final_x = base_pos[0] + 0.04
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