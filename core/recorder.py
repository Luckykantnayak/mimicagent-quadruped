from datetime import time

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
        self.root_pos = []
        self.joints = []
        self.time = []
        self.root_quat = []
        self.exp = []
        self.foot_positions = []

    def record(self, state):
        """
        state: dict returned by GaitControllerRunner.step()
        """
        time = state['time']

        pos = state["root_pos"]

        # MuJoCo quat: [w,x,y,z] → MimicKit: [x,y,z,w]
        q = state["root_quat"]
        quat = np.array([q[1], q[2], q[3], q[0]])
        exp = quat_to_exp_map(quat)

        joints = state["joints"]


        foot_positions_world = state["foot_positions_world"]


        frame = np.concatenate([pos, exp, joints])

        self.frames.append(frame)
        self.root_pos.append(pos)
        self.joints.append(joints)
        self.time.append(time)
        self.root_quat.append(q)
        self.exp.append(exp)
        self.foot_positions.append(foot_positions_world)

    def save(self, path):
        frames = np.vstack(self.frames)
        poses = np.vstack(self.root_pos)
        joints = np.vstack(self.joints)
        time = np.vstack(self.time)
        root_quat = np.vstack(self.root_quat)  # w, x, y, z
        exp = np.vstack(self.exp)
        foot_positions = np.stack(self.foot_positions, axis=0)
        # print(foot_positions.shape)
        motion = dict(
            fps=self.fps,
            loop_mode=1,
            frames=frames,
            time=time,
            joints=joints,
            root_quat=root_quat,
            root_pos=poses,
            exp=exp,
            foot_positions=foot_positions,
        )

        with open(path, "wb") as f:
            pickle.dump(motion, f)

        print(f" Saved MimicKit motion: {path}")
        print(f"   frames: {frames.shape}")
