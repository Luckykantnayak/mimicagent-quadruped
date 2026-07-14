import numpy as np
import pickle
from utils.math_utils import quat_to_exp_map
import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MimicKitRecorder:
    """
    Records the controller's per-step state and saves two artifacts:

    - <out>.pkl          : evaluation trajectory consumed by the agent's
                           unit tests (root_pos, root_quat, joints,
                           ee_positions arrays + fps/robot/ee_names metadata)
    - <out>_mimickit.pkl : MimicKit motion dict {fps, loop_mode, frames}
                           where frames = [root_pos(3), exp_map(3), joints(J)]
    """

    def __init__(self, fps, robot="", ee_names=None):
        self.fps = fps
        self.robot = robot
        self.ee_names = list(ee_names) if ee_names else []
        self.frames = []
        self.root_pos = []
        self.root_quat = []
        self.joints = []
        self.ee_positions = []

    def record(self, state):
        """
        state: dict returned by MotionControllerRunner.step()
        """

        pos = state["root_pos"]

        # MuJoCo quat: [w,x,y,z] → MimicKit: [x,y,z,w]
        q = state["root_quat"]
        quat = np.array([q[1], q[2], q[3], q[0]])

        exp = quat_to_exp_map(quat)
        joints = state["joints"]

        frame = np.concatenate([pos, exp, joints])
        self.frames.append(frame)

        self.root_pos.append(np.asarray(pos).copy())
        self.root_quat.append(np.asarray(q).copy())
        self.joints.append(np.asarray(joints).copy())
        self.ee_positions.append(np.asarray(state["ee_positions_world"]).copy())

    def save(self, path):
        path = Path(path)

        # ---- Evaluation trajectory (consumed by the agent's unit tests) ----
        traj = dict(
            root_pos=np.vstack(self.root_pos),          # (T, 3)
            root_quat=np.vstack(self.root_quat),        # (T, 4) [w,x,y,z]
            joints=np.vstack(self.joints),              # (T, J)
            ee_positions=np.stack(self.ee_positions),   # (T, N, 3)
            fps=self.fps,
            robot=self.robot,
            ee_names=self.ee_names,
        )
        with open(path, "wb") as f:
            pickle.dump(traj, f)
        print(f"✅ Saved eval trajectory: {path}")

        # ---- MimicKit motion ----
        frames = np.vstack(self.frames)
        motion = dict(
            fps=self.fps,
            loop_mode=1,
            frames=frames,
        )
        mimickit_path = path.with_name(path.stem + "_mimickit" + path.suffix)
        with open(mimickit_path, "wb") as f:
            pickle.dump(motion, f)
        print(f"✅ Saved MimicKit motion: {mimickit_path}")
        print(f"   frames: {frames.shape}")
