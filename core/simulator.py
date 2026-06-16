# core/simulator.py
import mujoco
import glfw
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MujocoSimulator:
    def __init__(self, model_path: str, init_position: str, sim_freq: int, base_body: str = "base_link"):
        """
        Args:
            model_path    : path to scene XML
            init_position : keyframe name in XML (e.g. "home")
            sim_freq      : simulation frequency in Hz
            base_body     : name of the root/base body for camera tracking
                            (comes from RobotConfig.base_body — no hardcoding needed)
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        if not glfw.init():
            raise RuntimeError("Could not initialise GLFW.")

        self.window = glfw.create_window(1280, 900, "Kinematic Viewer", None, None)
        glfw.make_context_current(self.window)

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.cam = mujoco.MjvCamera()

        # Camera tracking — uses the robot-config base body name
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        if body_id >= 0:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = body_id
        else:
            print(f"[WARN] Body '{base_body}' not found — switching camera to FREE mode")
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.trackbodyid = -1

        # self.cam.lookat[:] = [0, 0, 0.3]
        # self.cam.distance = 2
        # self.cam.azimuth = 90
        # self.cam.elevation = -20

        self.cam.lookat[:] = [0, 0, 0.9]  # higher for humanoid
        self.cam.distance = 4.0
        self.cam.azimuth = 90
        self.cam.elevation = -15

        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.dt = 1.0 / sim_freq

        self._reset(init_position)

    def _reset(self, init_position: str):
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, init_position)
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        else:
            print(f"[WARN] Keyframe '{init_position}' not found — using default reset.")
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_fwdPosition(self.model, self.data)

    def render(self):
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, self.pert,
            self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene,
        )
        mujoco.mjr_render(
            mujoco.MjrRect(0, 0, 1280, 900),
            self.scene, self.context,
        )
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        glfw.wait_events_timeout(self.dt)

    def close(self):
        glfw.terminate()