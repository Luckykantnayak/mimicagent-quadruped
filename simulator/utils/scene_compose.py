"""
utils/scene_compose.py

Compose a robot model + a robot-AGNOSTIC environment scene into one MjModel.

Scenes under `assets/scenes/` define ONLY the environment (floor, lights, textures,
objects) — no robot, no keyframe — so any robot can be dropped into any scene. But
MuJoCo loads a single file and resolves `<include>` and `meshdir` relative to the
MAIN file. So we write a tiny wrapper IN THE SCENE'S OWN DIRECTORY (where the robot
model's `meshdir="../robots/<name>/assets/"` and the env `<include>` both resolve),
load it, and delete it. The robot owns its `<keyframe home>`, so the composed model
resets to the robot's rest pose regardless of the environment.
"""

import os
import tempfile

import mujoco


def compose_model(robot_mjcf: str, scene_env: str) -> "mujoco.MjModel":
    """Return an MjModel of `robot_mjcf` placed in environment `scene_env`.

    Both are paths (repo-root-relative or absolute). The wrapper is written into the
    scene's directory so all relative asset paths resolve, then removed.
    """
    scene_env_abs = os.path.abspath(scene_env)
    scene_dir = os.path.dirname(scene_env_abs)
    rel_robot = os.path.relpath(os.path.abspath(robot_mjcf), scene_dir)
    rel_env = os.path.basename(scene_env_abs)
    wrapper = (
        '<mujoco model="composed">\n'
        f'  <include file="{rel_robot}"/>\n'
        f'  <include file="{rel_env}"/>\n'
        '</mujoco>\n'
    )
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="_compose_", dir=scene_dir)
    try:
        os.write(fd, wrapper.encode())
        os.close(fd)
        return mujoco.MjModel.from_xml_path(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
