from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_REAR_VAULT_MotionGenerator(BaseMotionGenerator):
    """
    Kinematic implementation of a rear‑driven vault.
    The base follows a smooth pitch trajectory and the legs execute
    coordinated push, lift, swing and landing motions.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        """
        Parameters
        ----------
        initial_foot_positions_body : dict
            Mapping leg name -> 3‑D position in body frame.
        """
        # Call base constructor
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Motion parameters
        self.freq = 1.0                     # cycle frequency (Hz)
        self.duty_front = 0.4               # front legs contact duration
        self.lift_start = 0.4              # phase when front lift begins
        self.lift_end = 0.55               # phase when front landing starts
        self.recovery_start = 0.55         # phase when rear recovery begins

        # Pitch trajectory parameters (radians)
        self.pitch_back = -0.25            # backward tilt during rear_shift
        self.pitch_forward = 0.35          # forward pitch during vault_push
        self.pitch_neutral = 0.0           # neutral orientation

        # Foot motion parameters
        self.step_length_front = 0.12      # forward swing distance for front legs
        self.step_height_front = 0.08      # lift height for front legs
        self.push_length_rear = -0.15     # rear leg push distance (negative forward)
        self.push_height_rear = 0.1       # push height for rear legs

        # Store previous pitch to compute angular velocity
        self.prev_pitch = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base pitch using a piecewise linear trajectory over the cycle.
        The angular velocity is derived from the change in pitch between steps.
        """
        # Determine desired pitch based on phase
        if phase < 0.20:                     # rear_shift
            desired_pitch = self.pitch_back * (phase / 0.20)
        elif phase < 0.40:                   # vault_push
            desired_pitch = self.pitch_back + \
                (self.pitch_forward - self.pitch_back) * ((phase - 0.20) / 0.20)
        elif phase < 0.55:                   # front_lift
            desired_pitch = self.pitch_forward + \
                (self.pitch_neutral - self.pitch_forward) * ((phase - 0.40) / 0.15)
        else:                                # front_landing + rear_recovery
            desired_pitch = self.pitch_neutral

        # Angular velocity about body x‑axis (pitch)
        pitch_rate = (desired_pitch - self.prev_pitch) / dt
        self.prev_pitch = desired_pitch

        # No linear motion in this skill
        self.vel_world = np.array([1.0, 0.0, 0.0])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])

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
        Compute the target foot position in body frame for a given leg.
        The motion is defined piecewise over the cycle.
        """
        base_pos = self.base_init_feet_pos[leg_name].copy()

        # Front legs (FL, FR)
        if leg_name.startswith(("FL", "FR")):

            # stance phase
            if phase < self.lift_start:
                base_pos[0] += 0.0
                base_pos[2] += 0.0

            # swing phase
            elif phase < self.lift_end:
                progress = (phase - self.lift_start) / (self.lift_end - self.lift_start)
                base_pos[0] += self.step_length_front * progress
                base_pos[2] += self.step_height_front * np.sin(np.pi * progress)

            # landing / stance
            else:
                base_pos[0] += self.step_length_front
                base_pos[2] += 0.0


        # Rear legs (RL, RR)
        else:  # Rear legs (RL, RR)

            # 0.00 – 0.20 : preload / slight backward extension
            if phase < 0.20:
                progress = phase / 0.20
                base_pos[0] -= 0.04 * progress
                base_pos[2] += 0.0

            # 0.20 – 0.40 : strong push (stance)
            elif phase < 0.40:
                progress = (phase - 0.20) / 0.20
                base_pos[0] += self.push_length_rear * progress
                base_pos[2] += self.push_height_rear * np.sin(np.pi * progress)

            # 0.40 – 0.55 : lift & recovery
            elif phase < self.lift_end:
                progress = (phase - 0.40) / (self.lift_end - 0.40)
                base_pos[0] += self.push_length_rear * (1.0 - progress)
                base_pos[2] += self.push_height_rear * np.sin(np.pi * (1.0 - progress))

            # 0.55 – 1.00 : stance reset
            else:
                base_pos[0] += 0.0
                base_pos[2] += 0.0

        return base_pos

    def compute_all_foot_positions_body_frame(self, phase):
        """
        Override to use the custom leg motion logic.
        """
        return {
            leg: self.compute_foot_position_body_frame(leg, phase)
            for leg in self.leg_names
        }