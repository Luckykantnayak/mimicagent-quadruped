# core/simulator.py
import mujoco
import glfw
import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class MujocoSimulator:
    def __init__(self, model_path, init_position, sim_freq):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Initialize GLFW window
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW.")

        self.window = glfw.create_window(1280, 900, "Kinematic Viewer", None, None)
        glfw.make_context_current(self.window)

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )

        self.cam.lookat[:] = [0, 0, 0.3]
        self.cam.distance = 2
        self.cam.azimuth, self.cam.elevation = 90, -20
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.dt = 1.0 / sim_freq

        self._reset(init_position)

    def _reset(self, init_position):
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, init_position)
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_fwdPosition(self.model, self.data)

    def render(self):
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, self.pert,
            self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        mujoco.mjr_render(
            mujoco.MjrRect(0, 0, 1280, 900),
            self.scene, self.context
        )
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        glfw.wait_events_timeout(self.dt)  # maintain ~200Hz wall-time update rate


    def close(self):
        glfw.terminate()
