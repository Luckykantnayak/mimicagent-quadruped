from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_FORWARD_FLIP_DASH_MotionGenerator(BaseMotionGenerator):
    """
    Continuous forward dash with aerial front flip.
    Implements base translation and incremental pitch rotation during flight,
    and leg tucking/extension per the motion plan.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        """
        Parameters
        ----------
        initial_foot_positions_body : dict
            Mapping leg name -> 3D position in BODY frame at rest.
        leg_names : list
            Ordered list of leg identifiers (e.g., ["FL", "FR", "RL", "RR"]).
        """
        # Call base initializer
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Store leg names
        self.leg_names = leg_names

        # Base motion parameters
        self.base_speed = 0.5               # forward speed (m/s)
        self.pitch_start_phase = 0.35
        self.pitch_end_phase = 0.65
        self.pitch_start_angle = 0.0        # radians
        self.pitch_end_angle = 2 * np.pi    # full front flip

        # Track previous pitch for incremental rotation
        self.prev_pitch_angle = 0.0

        # Leg motion definitions (piecewise linear segments)
        self.leg_segments = {}
        for leg in self.leg_names:
            segments = []
            if leg.startswith(("FL", "FR")):
                # Front legs: lift during flip
                segments.append((0.20, 0.35,
                                 self.base_init_feet_pos[leg],
                                 self.base_init_feet_pos[leg] + np.array([-0.05, 0.0, 0.0])))
                segments.append((0.35, 0.65,
                                 self.base_init_feet_pos[leg] + np.array([-0.05, 0.0, 0.0]),
                                 self.base_init_feet_pos[leg] + np.array([0.0, 0.0, -0.10])))
                segments.append((0.65, 1.00,
                                 self.base_init_feet_pos[leg] + np.array([0.0, 0.0, -0.10]),
                                 self.base_init_feet_pos[leg]))
            else:
                # Rear legs: stay on ground then lift slightly during flip
                segments.append((0.00, 0.20,
                                 self.base_init_feet_pos[leg],
                                 self.base_init_feet_pos[leg]))
                segments.append((0.20, 0.35,
                                 self.base_init_feet_pos[leg],
                                 self.base_init_feet_pos[leg]))
                segments.append((0.35, 0.65,
                                 self.base_init_feet_pos[leg],
                                 self.base_init_feet_pos[leg] + np.array([0.0, 0.0, -0.10])))
                segments.append((0.65, 1.00,
                                 self.base_init_feet_pos[leg] + np.array([0.0, 0.0, -0.10]),
                                 self.base_init_feet_pos[leg]))
            self.leg_segments[leg] = segments

    def update_base_motion(self, phase, dt):
        """
        Update base pose: constant forward speed and incremental pitch rotation during flip.
        """
        # Constant forward velocity
        self.vel_world = np.array([self.base_speed, 0.0, 0.0])
        self.omega_world = np.zeros(3)

        # Determine current pitch angle
        if self.pitch_start_phase <= phase < self.pitch_end_phase:
            progress = (phase - self.pitch_start_phase) / (self.pitch_end_phase - self.pitch_start_phase)
            pitch_angle = self.pitch_start_angle + progress * (self.pitch_end_angle - self.pitch_start_angle)
        else:
            pitch_angle = self.pitch_start_angle if phase < self.pitch_start_phase else self.pitch_end_angle

        # Incremental pitch rotation
        delta_pitch = pitch_angle - self.prev_pitch_angle
        self.prev_pitch_angle = pitch_angle

        # Convert incremental pitch to quaternion
        delta_quat = euler_to_quat(0.0, delta_pitch, 0.0)
        self.root_quat = quat_multiply(delta_quat, self.root_quat)

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
        Piecewise linear interpolation of foot position based on defined segments.
        """
        segs = self.leg_segments[leg_name]
        for (start, end, start_pos, end_pos) in segs:
            if start <= phase < end:
                progress = (phase - start) / (end - start)
                return start_pos + progress * (end_pos - start_pos)
        # Default to initial position if phase outside defined ranges
        return self.base_init_feet_pos[leg_name].copy()
