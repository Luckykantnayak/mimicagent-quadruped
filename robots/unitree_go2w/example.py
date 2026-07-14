from utils.math_utils import *
from motion.base import BaseMotionGenerator


class WalkingMotionGenerator(BaseMotionGenerator):
    """
    Straight-line diagonal trot (Unitree Go2W reference).

    End-effector (wheel-center) targets are authored in the BODY frame: each
    wheel sweeps backward during stance and arcs forward during swing. The
    base translates forward in the WORLD frame along its own heading, at a
    speed matched to the stance sweep so the planted wheels do not slip. No
    turning. The wheel joints are passive and never commanded.
    """

    def __init__(self, init_ee_pos, freq=1.0):
        super().__init__(init_ee_pos, freq=freq)

        self.duty = 0.6            # fraction of the cycle each wheel is in stance
        self.step_length = 0.12    # fore-aft wheel travel (m)
        self.step_height = 0.05    # peak swing lift (m)

        # Diagonal trot: FL+RR move together, FR+RL half a cycle later.
        self.phase_offsets = {
            ee: (0.0 if ee.startswith("FL") or ee.startswith("RR") else 0.5)
            for ee in self.ee_names
        }

        # Forward speed matched to the stance sweep -> stance wheels stay planted.
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
        pos = self.base_ee_pos[ee_name].copy()

        if ee_phase < self.duty:
            # Stance: wheel on the ground, sweeps backward under the body.
            progress = ee_phase / self.duty
            pos[0] -= self.step_length * (progress - 0.5)
        else:
            # Swing: wheel lifts and arcs forward to the next contact.
            progress = (ee_phase - self.duty) / (1.0 - self.duty)
            pos[0] += self.step_length * (progress - 0.5)
            pos[2] += self.step_height * np.sin(np.pi * progress)

        return pos
