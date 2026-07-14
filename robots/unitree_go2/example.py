from utils.math_utils import *
from motion.base import BaseMotionGenerator


class WalkingMotionGenerator(BaseMotionGenerator):
    """
    Straight-line diagonal trot (Unitree Go2 reference).

    End-effector targets are authored in the BODY frame: each foot sweeps
    backward during stance and arcs forward during swing. The base translates
    forward in the WORLD frame along its own heading, at a speed matched to
    the stance sweep so the planted feet do not slip. No turning.
    """

    def __init__(self, init_ee_pos, freq=1.0):
        super().__init__(init_ee_pos, freq=freq)

        self.duty = 0.6            # fraction of the cycle each foot is in stance
        self.step_length = 0.12    # fore-aft foot travel (m)
        self.step_height = 0.06    # peak swing-foot lift (m)

        # Diagonal trot: FL+RR move together, FR+RL half a cycle later.
        self.phase_offsets = {
            ee: (0.0 if ee.startswith("FL") or ee.startswith("RR") else 0.5)
            for ee in self.ee_names
        }

        # Forward speed matched to the stance sweep -> stance feet stay planted.
        self.forward_speed = self.step_length * self.freq / self.duty

    def update_base_motion(self, phase, dt):
        # Constant forward velocity along the body's heading, expressed in the
        # world frame; the base pose is integrated in the world frame.
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
            # Stance: foot planted, sweeps backward under the body.
            progress = ee_phase / self.duty
            foot[0] -= self.step_length * (progress - 0.5)
        else:
            # Swing: foot lifts and arcs forward to the next foothold.
            progress = (ee_phase - self.duty) / (1.0 - self.duty)
            foot[0] += self.step_length * (progress - 0.5)
            foot[2] += self.step_height * np.sin(np.pi * progress)

        return foot
