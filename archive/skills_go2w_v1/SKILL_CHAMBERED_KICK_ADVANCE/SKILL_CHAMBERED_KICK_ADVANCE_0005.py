from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CHAMBERED_KICK_ADVANCE_MotionGenerator(BaseMotionGenerator):
    """
    Chambered kick advance: sequential chamber-extend kicks per leg (RL → RR → FL → FR)
    for forward locomotion with martial-arts-inspired aesthetic.

    Phase structure:
      [0.0, 0.15]:  RL chamber
      [0.15, 0.3]:  RL extend (explosive)
      [0.3, 0.45]:  RR chamber
      [0.45, 0.6]:  RR extend (explosive)
      [0.6, 0.75]:  FL chamber-extend (controlled)
      [0.75, 0.9]:  FR chamber-extend (controlled)
      [0.9, 1.0]:   neutral settle
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.6

        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Time and base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Motion parameters - rear legs (powerful kicks)
        self.rear_chamber_height = 0.15
        self.rear_chamber_retract_x = 0.12
        self.rear_extend_forward_x = 0.18
        self.rear_extend_downward_z = -0.05

        # Motion parameters - front legs (controlled kicks)
        self.front_chamber_height = 0.10
        self.front_chamber_retract_x = 0.08
        self.front_extend_forward_x = 0.12
        self.front_extend_downward_z = -0.03

        # Base velocity parameters
        self.vx_low = 0.3
        self.vx_rear_surge = 1.5
        self.vx_front_surge = 0.8
        self.vx_settle = 0.2

        # Pitch compensation during explosive kicks
        self.pitch_rate_compensation = -0.3

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Explosive surges during rear leg extensions, moderate during front legs.
        """
        vx = 0.0
        pitch_rate = 0.0

        # RL chamber
        if 0.0 <= phase < 0.15:
            vx = self.vx_low
            pitch_rate = 0.0

        # RL extend (explosive)
        elif 0.15 <= phase < 0.3:
            progress = (phase - 0.15) / 0.15
            vx = self.vx_low + (self.vx_rear_surge - self.vx_low) * np.sin(np.pi * progress)
            pitch_rate = self.pitch_rate_compensation * np.sin(np.pi * progress)

        # RR chamber
        elif 0.3 <= phase < 0.45:
            vx = self.vx_low
            pitch_rate = 0.0

        # RR extend (explosive)
        elif 0.45 <= phase < 0.6:
            progress = (phase - 0.45) / 0.15
            vx = self.vx_low + (self.vx_rear_surge - self.vx_low) * np.sin(np.pi * progress)
            pitch_rate = self.pitch_rate_compensation * np.sin(np.pi * progress)

        # FL chamber-extend (controlled)
        elif 0.6 <= phase < 0.75:
            progress = (phase - 0.6) / 0.15
            vx = self.vx_low + (self.vx_front_surge - self.vx_low) * np.sin(np.pi * progress)
            pitch_rate = 0.0

        # FR chamber-extend (controlled)
        elif 0.75 <= phase < 0.9:
            progress = (phase - 0.75) / 0.15
            vx = self.vx_low + (self.vx_front_surge - self.vx_low) * np.sin(np.pi * progress)
            pitch_rate = 0.0

        # Neutral settle
        else:
            blend = (phase - 0.9) / 0.1
            vx = self.vx_low * (1.0 - blend) + self.vx_settle * blend
            pitch_rate = 0.0

        self.vel_world = np.array([vx, 0.0, 0.0])
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
        Compute foot trajectory in body frame for each leg based on phase.
        """
        foot = self.base_feet_pos_body[leg_name].copy()

        # RL: Rear Left
        if leg_name.startswith('RL'):
            foot = self._compute_RL_trajectory(foot, phase)

        # RR: Rear Right
        elif leg_name.startswith('RR'):
            foot = self._compute_RR_trajectory(foot, phase)

        # FL: Front Left
        elif leg_name.startswith('FL'):
            foot = self._compute_FL_trajectory(foot, phase)

        # FR: Front Right
        elif leg_name.startswith('FR'):
            foot = self._compute_FR_trajectory(foot, phase)

        return foot

    def _compute_RL_trajectory(self, foot, phase):
        """
        RL trajectory:
          [0.0, 0.15]: chamber (lift and retract)
          [0.15, 0.3]: extend (explosive forward-down)
          [0.3, 1.0]: stance (gradual return to nominal)
        """
        base_foot = self.base_feet_pos_body['RL'].copy()

        # Chamber phase
        if phase < 0.15:
            progress = phase / 0.15
            smooth = 0.5 * (1.0 - np.cos(np.pi * progress))
            foot[0] = base_foot[0] + self.rear_chamber_retract_x * smooth
            foot[2] = base_foot[2] + self.rear_chamber_height * smooth

        # Extend phase
        elif phase < 0.3:
            progress = (phase - 0.15) / 0.15
            smooth = 0.5 * (1.0 - np.cos(np.pi * progress))

            # Transition from chambered to extended
            chamber_x = base_foot[0] + self.rear_chamber_retract_x
            chamber_z = base_foot[2] + self.rear_chamber_height
            extend_x = base_foot[0] + self.rear_extend_forward_x
            extend_z = base_foot[2] + self.rear_extend_downward_z

            foot[0] = chamber_x + (extend_x - chamber_x) * smooth
            foot[2] = chamber_z + (extend_z - chamber_z) * smooth

        # Stance phase - gradual return to nominal
        else:
            progress = (phase - 0.3) / 0.7
            smooth = min(1.0, progress * 1.5)
            extend_x = base_foot[0] + self.rear_extend_forward_x
            extend_z = base_foot[2] + self.rear_extend_downward_z

            foot[0] = extend_x + (base_foot[0] - extend_x) * smooth
            foot[2] = extend_z + (base_foot[2] - extend_z) * smooth

        return foot

    def _compute_RR_trajectory(self, foot, phase):
        """
        RR trajectory:
          [0.0, 0.3]: stance
          [0.3, 0.45]: chamber
          [0.45, 0.6]: extend
          [0.6, 1.0]: stance
        """
        base_foot = self.base_feet_pos_body['RR'].copy()

        # Early stance
        if phase < 0.3:
            foot = base_foot.copy()

        # Chamber phase
        elif phase < 0.45:
            progress = (phase - 0.3) / 0.15
            smooth = 0.5 * (1.0 - np.cos(np.pi * progress))
            foot[0] = base_foot[0] + self.rear_chamber_retract_x * smooth
            foot[2] = base_foot[2] + self.rear_chamber_height * smooth

        # Extend phase
        elif phase < 0.6:
            progress = (phase - 0.45) / 0.15
            smooth = 0.5 * (1.0 - np.cos(np.pi * progress))

            chamber_x = base_foot[0] + self.rear_chamber_retract_x
            chamber_z = base_foot[2] + self.rear_chamber_height
            extend_x = base_foot[0] + self.rear_extend_forward_x
            extend_z = base_foot[2] + self.rear_extend_downward_z

            foot[0] = chamber_x + (extend_x - chamber_x) * smooth
            foot[2] = chamber_z + (extend_z - chamber_z) * smooth

        # Stance phase - return to nominal
        else:
            progress = (phase - 0.6) / 0.4
            smooth = min(1.0, progress * 1.5)
            extend_x = base_foot[0] + self.rear_extend_forward_x
            extend_z = base_foot[2] + self.rear_extend_downward_z

            foot[0] = extend_x + (base_foot[0] - extend_x) * smooth
            foot[2] = extend_z + (base_foot[2] - extend_z) * smooth

        return foot

    def _compute_FL_trajectory(self, foot, phase):
        """
        FL trajectory:
          [0.0, 0.6]: stance
          [0.6, 0.75]: chamber-extend (combined, controlled)
          [0.75, 1.0]: stance
        """
        base_foot = self.base_feet_pos_body['FL'].copy()

        # Early stance
        if phase < 0.6:
            foot = base_foot.copy()

        # Chamber-extend combined phase
        elif phase < 0.75:
            total_duration = 0.15
            progress = (phase - 0.6) / total_duration

            # First half: chamber
            if progress < 0.5:
                chamber_progress = progress / 0.5
                smooth = 0.5 * (1.0 - np.cos(np.pi * chamber_progress))
                foot[0] = base_foot[0] + self.front_chamber_retract_x * smooth
                foot[2] = base_foot[2] + self.front_chamber_height * smooth

            # Second half: extend
            else:
                extend_progress = (progress - 0.5) / 0.5
                smooth = 0.5 * (1.0 - np.cos(np.pi * extend_progress))

                chamber_x = base_foot[0] + self.front_chamber_retract_x
                chamber_z = base_foot[2] + self.front_chamber_height
                extend_x = base_foot[0] + self.front_extend_forward_x
                extend_z = base_foot[2] + self.front_extend_downward_z

                foot[0] = chamber_x + (extend_x - chamber_x) * smooth
                foot[2] = chamber_z + (extend_z - chamber_z) * smooth

        # Late stance - return to nominal
        else:
            progress = (phase - 0.75) / 0.25
            smooth = min(1.0, progress * 2.0)
            extend_x = base_foot[0] + self.front_extend_forward_x
            extend_z = base_foot[2] + self.front_extend_downward_z

            foot[0] = extend_x + (base_foot[0] - extend_x) * smooth
            foot[2] = extend_z + (base_foot[2] - extend_z) * smooth

        return foot

    def _compute_FR_trajectory(self, foot, phase):
        """
        FR trajectory:
          [0.0, 0.75]: stance
          [0.75, 0.9]: chamber-extend (combined, controlled)
          [0.9, 1.0]: stance
        """
        base_foot = self.base_feet_pos_body['FR'].copy()

        # Early stance
        if phase < 0.75:
            foot = base_foot.copy()

        # Chamber-extend combined phase
        elif phase < 0.9:
            total_duration = 0.15
            progress = (phase - 0.75) / total_duration

            # First half: chamber
            if progress < 0.5:
                chamber_progress = progress / 0.5
                smooth = 0.5 * (1.0 - np.cos(np.pi * chamber_progress))
                foot[0] = base_foot[0] + self.front_chamber_retract_x * smooth
                foot[2] = base_foot[2] + self.front_chamber_height * smooth

            # Second half: extend
            else:
                extend_progress = (progress - 0.5) / 0.5
                smooth = 0.5 * (1.0 - np.cos(np.pi * extend_progress))

                chamber_x = base_foot[0] + self.front_chamber_retract_x
                chamber_z = base_foot[2] + self.front_chamber_height
                extend_x = base_foot[0] + self.front_extend_forward_x
                extend_z = base_foot[2] + self.front_extend_downward_z

                foot[0] = chamber_x + (extend_x - chamber_x) * smooth
                foot[2] = chamber_z + (extend_z - chamber_z) * smooth

        # Final stance - settle at nominal
        else:
            progress = (phase - 0.9) / 0.1
            smooth = min(1.0, progress * 3.0)
            extend_x = base_foot[0] + self.front_extend_forward_x
            extend_z = base_foot[2] + self.front_extend_downward_z

            foot[0] = extend_x + (base_foot[0] - extend_x) * smooth
            foot[2] = extend_z + (base_foot[2] - extend_z) * smooth

        return foot