from utils.math_utils import *
from motion.base import BaseMotionGenerator


class WalkingMotionGenerator(BaseMotionGenerator):
    """
    Straight-line biped walk (humanoid reference).

    The two feet (the only IK end-effectors) alternate stance and swing half a
    cycle apart, each authored in the BODY frame: the stance foot sweeps
    backward under the pelvis while the swing foot lifts and steps forward.
    The pelvis translates forward in the WORLD frame at a speed matched to the
    stance sweep, so the stance foot stays planted. The arms are lowered to
    hang at the sides and swing fore-aft via joint overrides (they are not IK
    end-effectors), anti-phase to each other and contralateral to the legs, as
    in a natural gait. No turning.
    """

    def __init__(self, init_ee_pos, freq=1.0):
        super().__init__(init_ee_pos, freq=freq)

        self.duty = 0.6
        self.step_length = 0.30    # fore-aft foot travel (m)
        self.step_height = 0.10    # peak swing-foot lift (m)

        # Left and right feet are half a cycle out of phase.
        self.phase_offsets = {
            ee: (0.0 if "right" in ee else 0.5) for ee in self.ee_names
        }

        # Arm posture (the zero pose is a T-pose, so adduct the shoulders to
        # bring the arms down to the sides, then swing them fore-aft).
        self.arm_down = 1.45       # shoulder_x adduction: arm hangs at the side
        self.arm_swing_amp = 0.4   # shoulder_z fore-aft swing amplitude (rad)
        self.elbow_bend = 0.35     # slight constant elbow flexion (rad)

        # Forward speed matched to the stance sweep -> stance foot stays planted.
        self.forward_speed = self.step_length * self.freq / self.duty

    def update_base_motion(self, phase, dt):
        # Constant forward velocity along the body's heading, expressed in the
        # world frame; the pelvis pose is integrated in the world frame.
        vel_world = quat_to_rotation_matrix(self.root_quat) @ np.array(
            [self.forward_speed, 0.0, 0.0]
        )
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos, self.root_quat, vel_world, np.zeros(3), dt
        )

    def compute_ee_position_body_frame(self, ee_name, phase):
        ee_phase = (phase + self.phase_offsets[ee_name]) % 1.0
        foot = self.base_ee_pos[ee_name].copy()

        if ee_phase < self.duty:
            # Stance: foot planted, sweeps backward under the pelvis.
            progress = ee_phase / self.duty
            foot[0] -= self.step_length * (progress - 0.5)
        else:
            # Swing: foot lifts and steps forward.
            progress = (ee_phase - self.duty) / (1.0 - self.duty)
            foot[0] += self.step_length * (progress - 0.5)
            foot[2] += self.step_height * np.sin(np.pi * progress)

        return foot

    def compute_joint_overrides(self, phase):
        # Shoulders are adducted (shoulder_x) so the arms hang at the sides;
        # the fore-aft swing is shoulder_z. Because the left shoulder mirrors
        # the right, the same shoulder_z value swings the two arms in
        # anti-phase — right arm back while the right (leading) leg is forward,
        # i.e. contralateral. Elbows carry a slight constant bend.
        swing = -self.arm_swing_amp * np.cos(2.0 * np.pi * phase)
        return {
            "right_shoulder_x": self.arm_down,
            "left_shoulder_x": -self.arm_down,
            "right_shoulder_z": swing,
            "left_shoulder_z": swing,
            "right_elbow": self.elbow_bend,
            "left_elbow": -self.elbow_bend,
        }
