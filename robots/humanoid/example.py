import numpy as np
from utils.math_utils import *
from motion.base import BaseMotionGenerator


class WalkingMotionGenerator(BaseMotionGenerator):
    """
    Straight-line biped walk (humanoid reference).

    THIS skill picks two of the three control channels. The choice is per limb
    per phase and follows from COUPLING — what actually determines where the limb
    goes — not from the morphology. A different skill on this same robot makes a
    different choice (a mantle puts the hands on WORLD targets; a squat-and-grip
    puts them on the object). Here:

      * FEET use BODY-frame IK — authored relative to the torso, because a swing
        foot's target is fixed by the torso, not by the world. They alternate
        stance/swing half a cycle apart and IK solves the legs.

      * ARMS are JOINT-driven — the hands are NOT IK targets. Their body-frame
        target is None (the base-class default), so IK skips them, and the
        shoulders swing fore-aft through compute_joint_overrides, each arm
        contralateral to the same-side leg. The arm's SHAPE is the point and its
        fingertip is unconstrained, so driving the shoulder ANGLE keeps the swing
        — where IK on a fingertip target would dissolve it into whatever reaches
        that point.

      * WORLD-frame IK is unused here: on flat ground nothing pins a foot or hand
        to a fixed world location. That is a property of this skill, not of the
        robot.

    The pelvis translates forward in the world frame, matched to the stance sweep
    so the stance foot stays planted; the head turns slightly with the stride.
    No turning.
    """

    def __init__(self, init_ee_pos, freq=1.0):
        super().__init__(init_ee_pos, freq=freq)

        self.duty = 0.6
        self.step_length = 0.30     # fore-aft foot travel (m)
        self.step_height = 0.10     # peak swing-foot lift (m)
        self.arm_swing = 0.5        # fore-aft shoulder swing (radians)

        # feet alternate half a cycle apart
        self.foot_offsets = {"right_foot": 0.0, "left_foot": 0.5}

        # forward speed matched to the stance sweep -> stance foot stays planted
        self.forward_speed = self.step_length * self.freq / self.duty

    def update_base_motion(self, phase, dt):
        # constant forward velocity along the heading, integrated in the world frame
        vel_world = quat_to_rotation_matrix(self.root_quat) @ np.array(
            [self.forward_speed, 0.0, 0.0]
        )
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos, self.root_quat, vel_world, np.zeros(3), dt
        )

    def compute_ee_position_body_frame(self, ee_name, phase):
        # Only the FEET are IK-driven. Returning None for the hands tells IK to
        # skip them — the arms are shaped by compute_joint_overrides below.
        if not ee_name.endswith("foot"):
            return None

        pos = self.base_ee_pos[ee_name].copy()
        ee_phase = (phase + self.foot_offsets[ee_name]) % 1.0
        if ee_phase < self.duty:
            # stance: foot planted, sweeps backward under the pelvis
            progress = ee_phase / self.duty
            pos[0] -= self.step_length * (progress - 0.5)
        else:
            # swing: foot lifts and steps forward
            progress = (ee_phase - self.duty) / (1.0 - self.duty)
            pos[0] += self.step_length * (progress - 0.5)
            pos[2] += self.step_height * np.sin(np.pi * progress)
        return pos

    def compute_joint_overrides(self, phase):
        # Arms swing fore-aft on the shoulder_z (flexion) joint. Both shoulders
        # take the SAME signed value: because the left arm mirrors the right,
        # one goes forward while the other goes back — a natural anti-phase swing,
        # contralateral to the legs. The head turns subtly with the stride.
        swing = self.arm_swing * np.cos(2.0 * np.pi * phase)
        return {
            "right_shoulder_z": swing,
            "left_shoulder_z": swing,
            "neck_z": 0.10 * np.sin(2.0 * np.pi * phase),
        }
