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
    def __init__(self, fps):
        self.fps = fps
        self.frames = []

    def record(self, state):
        """
        state: dict returned by GaitControllerRunner.step()
        """

        pos = state["root_pos"]

        # MuJoCo quat: [w,x,y,z] → MimicKit: [x,y,z,w]
        q = state["root_quat"]
        quat = np.array([q[1], q[2], q[3], q[0]])

        exp = quat_to_exp_map(quat)
        joints = state["joints"]

        frame = np.concatenate([pos, exp, joints])

        self.frames.append(frame)

    def save(self, path):
        frames = np.vstack(self.frames)

        motion = dict(
            fps=self.fps,
            loop_mode=1,
            frames=frames
        )

        with open(path, "wb") as f:
            pickle.dump(motion, f)

        print(f"✅ Saved MimicKit motion: {path}")
        print(f"   frames: {frames.shape}")
