from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising Spiral Jump: Vertical jump with continuous yaw rotation and
    sequential spiral leg extension during aerial phase.
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 1.0

        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        self.compression_depth = 0.06
        self.jump_height = 0.42
        self.total_yaw_rotation = 2.0 * np.pi

        self.max_radial_extension_x = 1.16
        self.max_radial_extension_y = 1.10

        self.leg_extension_factors = {
            leg_names[0]: 1.0,
            leg_names[1]: 1.0,
            leg_names[2]: 0.90,
            leg_names[3]: 0.90,
        }

        self.leg_tuck_factors = {
            leg_names[0]: 1.0,
            leg_names[1]: 1.0,
            leg_names[2]: 0.85,
            leg_names[3]: 0.85,
        }

        self.spiral_phase_offsets = {
            leg_names[0]: 0.0,
            leg_names[1]: 0.05,
            leg_names[2]: 0.15,
            leg_names[3]: 0.10,
        }

        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)


    def update_base_motion(self, phase, dt):

        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0

        if phase < 0.2:

            progress = phase / 0.2
            vz = -0.5 * (1.0 - progress)

        elif phase < 0.4:

            progress = (phase - 0.2) / 0.2
            vz = 1.9 * (1.0 - progress**0.5)
            yaw_rate = self.total_yaw_rotation / (0.8 * (1.0 / self.freq))

        elif phase < 0.8:

            progress = (phase - 0.4) / 0.4
            vz = 1.5 * (1.0 - 2.0 * progress)
            yaw_rate = self.total_yaw_rotation / (0.8 * (1.0 / self.freq))

        elif phase < 0.92:

            progress = (phase - 0.8) / 0.12
            vz = -0.9 * (1.0 - progress)
            yaw_rate = self.total_yaw_rotation / (0.8 * (1.0 / self.freq)) * (1.0 - 0.5 * progress)

        else:

            vz = 0.0
            yaw_rate = 0.0

        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])

        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )


    def compute_foot_position_body_frame(self, leg_name, phase):

        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()

        leg_factor = self.leg_extension_factors[leg_name]
        tuck_factor = self.leg_tuck_factors[leg_name]
        spiral_offset = self.spiral_phase_offsets[leg_name]


        if phase < 0.2:

            progress = phase / 0.2
            compression_factor = np.sin(progress * np.pi / 2)

            foot[0] *= (1.0 - 0.24 * compression_factor)
            foot[1] *= (1.0 - 0.24 * compression_factor)
            foot[2] += self.compression_depth * compression_factor


        elif phase < 0.4:

            progress = (phase - 0.2) / 0.2
            extension_factor = progress**1.3

            foot[0] = base_pos[0] * (0.76 + 0.34 * extension_factor)
            foot[1] = base_pos[1] * (0.76 + 0.34 * extension_factor)

            foot[2] = base_pos[2] + self.compression_depth * (1.0 - extension_factor) - 0.03 * extension_factor


        elif phase < 0.6:

            aerial_phase = phase - 0.4
            leg_extension_start = spiral_offset
            leg_extension_duration = 0.18

            if aerial_phase < leg_extension_start:

                foot[0] = base_pos[0] * 1.08
                foot[1] = base_pos[1] * 1.08
                foot[2] = base_pos[2] + 0.01

            elif aerial_phase < leg_extension_start + leg_extension_duration:

                ext_progress = (aerial_phase - leg_extension_start) / leg_extension_duration
                ext_curve = ext_progress**1.4

                radial_mult_x = 1.08 + (self.max_radial_extension_x - 1.08) * ext_curve * leg_factor
                radial_mult_y = 1.08 + (self.max_radial_extension_y - 1.08) * ext_curve * leg_factor

                foot[0] = base_pos[0] * radial_mult_x
                foot[1] = base_pos[1] * radial_mult_y
                foot[2] = base_pos[2] + 0.01

            else:

                foot[0] = base_pos[0] * (1.08 + (self.max_radial_extension_x - 1.08) * leg_factor)
                foot[1] = base_pos[1] * (1.08 + (self.max_radial_extension_y - 1.08) * leg_factor)
                foot[2] = base_pos[2] + 0.01


        elif phase < 0.75:

            progress = (phase - 0.6) / 0.15
            tuck_curve = np.sin(progress * np.pi / 2)

            foot[0] = base_pos[0] * 1.08
            foot[1] = base_pos[1] * 1.08
            foot[2] = base_pos[2] + (0.05 * tuck_curve) * tuck_factor


        elif phase < 0.88:

            progress = (phase - 0.75) / 0.13

            foot[0] = base_pos[0] * (1.08 - 0.08 * progress)
            foot[1] = base_pos[1] * (1.08 - 0.08 * progress)

            foot[2] = base_pos[2] + (0.05 * (1.0 - progress)) * tuck_factor


        elif phase < 0.96:

            progress = (phase - 0.88) / 0.08

            foot[0] = base_pos[0]
            foot[1] = base_pos[1]

            clearance = 0.015 * (1.0 - progress)
            foot[2] = base_pos[2] + clearance


        else:

            foot[0] = base_pos[0]
            foot[1] = base_pos[1]
            foot[2] = base_pos[2] + 0.003


        return foot