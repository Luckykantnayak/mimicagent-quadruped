# core/simulator.py
import mujoco
import glfw
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MujocoSimulator:
    def __init__(self, model_path: str, init_position: str, sim_freq: int,
                 base_body: str = "base_link", camera: dict = None,
                 headless: bool = False, robot_mjcf: str = None):
        """
        Args:
            model_path    : path to the environment scene XML (floor + objects only;
                            robot-agnostic). Composed with `robot_mjcf` at load.
            init_position : keyframe name in XML (e.g. "home")
            sim_freq      : simulation frequency in Hz
            base_body     : name of the root/base body for camera tracking
                            (comes from RobotSpec.base_body — no hardcoding needed)
            camera        : optional dict {lookat, distance, azimuth, elevation}
                            from RobotSpec.camera
            robot_mjcf    : path to the robot model XML (RobotSpec.mjcf). The scene
                            and robot are composed into one model at load. If None
                            (legacy full-scene XML), the scene is loaded directly.
        """
        if robot_mjcf:
            from utils.scene_compose import compose_model
            self.model = compose_model(robot_mjcf, model_path)
        else:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.headless = headless
        self.dt = 1.0 / sim_freq

        # Headless (agent pipeline): skip glfw/GL entirely — no window, no render,
        # no real-time throttle, no DISPLAY needed. Only the recorded pkl matters.
        if not headless:
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

            # Camera framing — per-robot values come from the robot spec
            camera = camera or {}
            self.cam.lookat[:] = camera.get("lookat", [0, 0, 0.5])
            self.cam.distance = camera.get("distance", 3.0)
            self.cam.azimuth = camera.get("azimuth", 90)
            self.cam.elevation = camera.get("elevation", -20)

            self.opt = mujoco.MjvOption()
            self.pert = mujoco.MjvPerturb()
        else:
            self.window = None

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
        if self.headless:
            return
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

    def should_close(self):
        return False if self.headless else glfw.window_should_close(self.window)

    def close(self):
        if not self.headless:
            glfw.terminate()