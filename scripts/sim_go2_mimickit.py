import mujoco
import numpy as np
import glfw
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import pickle
from pathlib import Path
import argparse
from utils import *
from gaits import *
import yaml
from enum import Enum
import pickle

# This copies MimicKit's enum
class LoopMode(Enum):
    CLAMP = 0
    WRAP = 1


class MultiLinkGradientDescentIK:
    """Multi-link Gradient Descent IK solver using stacked Jacobians"""

    def __init__(self, model, data, body_names, step_size=0.01, tol=1e-3):
        self.model = model
        self.data = data
        self.body_names = body_names
        self.step_size = step_size
        self.tol = tol

        # Get body IDs
        self.body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]

        # Get site IDs (foot tips)
        self.site_ids = []
        for name in body_names:
            # Guess foot site name
            site_name = name.replace("_calf", "_foot")
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                print(f"⚠️  Site {site_name} not found; using body {name} position")
                sid = None
            self.site_ids.append(sid)

        # Collect controllable DOFs (joint indices)
        self.dof_ids = self._get_leg_dofs()

        print(f"IK initialized: {len(self.dof_ids)} DOFs, {len(self.site_ids)} sites")

    def _get_leg_dofs(self):
        dof_ids = []
        for leg in ["FL", "FR", "RL", "RR"]:
            for joint in ["hip", "thigh", "calf"]:
                jn = f"{leg}_{joint}_joint"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    dof_ids.append(self.model.jnt_dofadr[jid])
        return np.array(dof_ids, dtype=int)

    def get_foot_positions(self):
        pos = []
        for sid, bid in zip(self.site_ids, self.body_ids):
            if sid is not None:
                pos.append(self.data.site_xpos[sid].copy())
            else:
                pos.append(self.data.xpos[bid].copy())
        return np.array(pos)

    def damped_least_squares(self, J, dx, damping=1e-3):
        H = J.T @ J + (damping**2) * np.eye(J.shape[1])
        dq = np.linalg.solve(H, J.T @ dx)
        return dq

    def calculate(self, target_feet_pos, max_iter=500, damping=1e-3, debug=False):
        mujoco.mj_fwdPosition(self.model, self.data)
        dq = np.zeros(self.model.nv)

        for it in range(max_iter):
            current = self.get_foot_positions()
            err = (target_feet_pos - current).flatten()
            avg_err = np.linalg.norm(err) / len(self.site_ids)
            if avg_err < self.tol:
                break

            # Build stacked Jacobian
            J_list = []
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            for sid in self.site_ids:
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
                J_list.append(jacp[:, self.dof_ids].copy())
            J = np.vstack(J_list)

            dq_local = self.step_size * self.damped_least_squares(J, err, damping)
            dq[self.dof_ids] = dq_local
            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1)
            mujoco.mj_fwdPosition(self.model, self.data)

        if debug:
            print(f"[IK] Finished in {it} iters, avg_err={avg_err*1000:.2f} mm")

        return avg_err


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    # ---- Parse config path only ----
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default="../config/config.yaml",
        help="Path to YAML config"
    )
    cfg_args, remaining_argv = config_parser.parse_known_args()

    # ---- Load YAML defaults ----
    with open(cfg_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- Main parser (NO defaults here) ----
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str)
    parser.add_argument("--robot", type=str)
    parser.add_argument("--record", type=str2bool)
    parser.add_argument("--plot", type=str2bool)
    parser.add_argument("--record_cycles", type=int)

    parser.add_argument("--init_position", type=str)
    parser.add_argument("--skating_style", type=str)
    parser.add_argument("--base_velocity", type=float)
    parser.add_argument("--yaw_rate", type=float)
    parser.add_argument("--step_length", type=float)
    parser.add_argument("--step_height", type=float)
    parser.add_argument("--gait_freq", type=float)
    parser.add_argument("--sim_freq", type=float)

    # ---- YAML → defaults ----
    parser.set_defaults(**cfg)

    # ---- Parse remaining CLI args (override YAML) ----
    args = parser.parse_args(remaining_argv)

    return args


# ============================================================
#                MAIN FUNCTION
# ============================================================
def main():
    args = get_args()
    # print(vars(args))

    robot = args.robot
    mode = args.mode
    record = args.record
    record_cycles = args.record_cycles
    init_position = args.init_position
    skating_style = args.skating_style

    # Load model (pure MJCF, no physics necessary)
    model = mujoco.MjModel.from_xml_path(f"../assets/unitree_{robot}/scene.xml")
    data = mujoco.MjData(model)

    # Initialize GLFW window
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW.")
    window = glfw.create_window(1280, 900, "Go2 Kinematic IK Demo", None, None)
    glfw.make_context_current(window)

    # Visualization setup
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
    )

    cam.lookat[:] = [0, 0, 0.3]
    cam.distance = 2
    cam.azimuth, cam.elevation = 90, -20
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    
    # Reset to home pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, init_position)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    
    mujoco.mj_fwdPosition(model, data)

    # Initialize IK
    body_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    ik = MultiLinkGradientDescentIK(model, data, body_names)

    initial_feet_pos = ik.get_foot_positions()
    print("Initial foot positions:")
    for i, n in enumerate(body_names):
        print(f"  {n}: {initial_feet_pos[i]}")

    frame = 0
    
    # Convert to dictionary for easy access
    base_feet = {n: initial_feet_pos[i].copy() for i, n in enumerate(body_names)}
        
    # --------------------------------
    # Live Matplotlib setup
    # --------------------------------
    if args.plot:
        plt.ion()
        fig, (ax_z, ax_x) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        colors = ["r", "g", "b", "m"]
        lines_z, lines_x = [], []

        # Buffers (fixed-size deques for real-time)
        buffer_len = 100
        total_time = 0.5
        time_buffer = deque(np.linspace(0, total_time, buffer_len), maxlen=buffer_len)
        pos_z_buffers = [deque(np.zeros(buffer_len), maxlen=buffer_len) for _ in body_names]
        pos_x_buffers = [deque(np.zeros(buffer_len), maxlen=buffer_len) for _ in body_names]

        # Z plot setup
        for i, name in enumerate(body_names):
            line_z, = ax_z.plot(time_buffer, pos_z_buffers[i], color=colors[i], label=f"{name} Z")
            lines_z.append(line_z)
        ax_z.set_xlim(0, total_time)
        ax_z.set_ylim(-0.4, 0.4)
        ax_z.set_ylabel("Foot Z height (m)")
        ax_z.legend(loc="upper right")

        # X plot setup
        for i, name in enumerate(body_names):
            line_x, = ax_x.plot(time_buffer, pos_x_buffers[i], color=colors[i], linestyle="--", label=f"{name} X")
            lines_x.append(line_x)
        ax_x.set_xlim(0, total_time)
        ax_x.set_ylim(-0.5, 0.5)
        ax_x.set_xlabel("Time (s)")
        ax_x.set_ylabel("Foot X position (m)")
        ax_x.legend(loc="upper right")

        plt.tight_layout()
        plt.show(block=False)
    
    # --------------------------------
    # Define gait settings
    # --------------------------------
    base_velocity = args.base_velocity
    step_length = args.step_length
    step_height = args.step_height
    gait_freq = args.gait_freq
    yaw_rate = args.yaw_rate

    if mode=="walking" or mode=="static":
        print("Walking Gait Controller Initialized")
        gait = WalkingGaitController(
                base_init_feet_pos=base_feet,
                freq=gait_freq,
                duty_ratio=0.75,
                step_length=step_length,
                step_height=step_height
            )
    elif  mode=="skating":
        print("Skating Gait Controller Initialized")
        gait = SkatingGaitController(
                    base_init_feet_pos=base_feet,
                    freq=gait_freq,
                    step_length=step_length,
                    step_height=step_height,
                    style=skating_style
                )

    elif mode == "backflip":
        print("Backflip Controller Initialized")
        gait = BackflipGaitController(
            base_init_feet_pos=base_feet,
            total_time=record_cycles / gait_freq,  # duration of backflip
            step_height=0.05
        )

    # --------------------------------
    # One gait cycle duration
    # --------------------------------
    T_cycle = 1.0 / gait_freq

    # --------------------------------
    # Simulation timing
    # --------------------------------
    sim_freq = args.sim_freq  # args.sim_freq = 200         # Hz
    dt = 1.0 / sim_freq
    FPS = int(sim_freq)       # stored in MimicKit file

    # --------------------------------
    # Storage
    # --------------------------------
    frames_qpos = []          # store raw MuJoCo qpos
    frames_root_pos = []
    frames_root_exp = []
    frames_joint = []
    frames_foot_body = []
    frames_phase = []

    t = 0.0
    start_time = 0.5
    end_time = record_cycles*T_cycle + start_time
    # -------------------------------
    # Main simulation loop
    # -------------------------------
    while not glfw.window_should_close(window):
        target_feet_pos = initial_feet_pos.copy()
        qpos = data.qpos.copy()
        _, _, yaw = quat_to_euler(qpos[3:7])
        gait.set_base_init_feet_pos(vx=base_velocity, yaw=yaw, dt=dt, yaw_rate=yaw_rate)

        if mode == "backflip":
            data.qpos[2] = gait.com_height(t)  # base height dynamically
            data.qpos[3:7] = gait.base_orientation(t)  # rotate base

        for i, n in enumerate(body_names):
            if mode=="skating":               
                target_feet_pos[i] = gait.foot_target(n, t)
                # print(f"Leg {i} : ", target_feet_pos[i])

            elif mode=="walking":
                target_feet_pos[i] = gait.foot_target(n, t, mode="moving")

            elif mode=="backflip":
                target_feet_pos[i] = gait.foot_target(n, t)

            else:
                target_feet_pos[i] = gait.foot_target(n, t, mode="static")


        err = ik.calculate(target_feet_pos, debug=(frame % 60 == 0))

        # --- Collect data for plot ---
        if args.plot: 
            feet = ik.get_foot_positions()  # shape (4, 3)

            for i in range(4):
                pos_z_buffers[i].append(feet[i, 2])
                pos_x_buffers[i].append(feet[i, 0])

            # --- Update plot every few frames ---
            if  frame % 2 == 0:
                for i in range(4):
                    lines_z[i].set_ydata(pos_z_buffers[i])
                    lines_x[i].set_ydata(pos_x_buffers[i])
                plt.pause(0.001)
        
        
        # --------------------------------
        # Collect data
        # --------------------------------
        qpos = data.qpos.copy()
        frames_qpos.append(qpos)

        # Root pose
        root_pos = qpos[0:3] + delta_vector(vx=base_velocity, theta=yaw, dt=dt, yaw_rate=yaw_rate)
        root_orientation = integrate_yaw(qpos[3:7], yaw_rate, dt)

        # Convert MuJoCo (w,x,y,z) → MimicKit order (x,y,z,w)
        mj_quat = qpos[3:7]        # [w,x,y,z]
        root_quat = np.array([mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]])
        root_exp = quat_to_exp_map(root_quat)
        
        if record and t >= start_time:
            frames_root_pos.append(root_pos)
            frames_root_exp.append(root_exp)
            # Joint angles (everything after the floating base)
            frames_joint.append(qpos[7:].copy())

        # --- Render simulation ---
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, 1280, 900), scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

        if record and t >= end_time:  
            print(f"Recorded {record_cycles} Gait Cycle")
            break

        # --- Advance time with fixed dt ---
        frame += 1
        t += dt
        data.qpos[0:3] = root_pos
        data.qpos[3:7] = root_orientation
        glfw.wait_events_timeout(dt)  # maintain ~200Hz wall-time update rate

    glfw.terminate()
    # ============================================================
    #        BUILD MimicKit Frames: [pos(3), expmap(3), joints]
    # ============================================================
    frames = []
    for pos, exp, joint in zip(frames_root_pos, frames_root_exp, frames_joint):
        frame = np.concatenate([pos, exp, joint])
        frames.append(frame)

    frames = np.vstack(frames)   # shape [T, D]

    # ============================================================
    #        SAVE MimicKit motion file
    # ============================================================
    motion = dict(
        fps=FPS,
        loop_mode=1,  # WRAP
        frames=frames
        # phase=np.array(frames_phase),         # optional but useful
        # foot_body=np.array(frames_foot_body)  # optional debugging
    )
    if mode=="skating":
        save_path = Path(f"../data/{robot}_{skating_style}_{mode}_gait_mimickit.pkl")
    else:
        save_path = Path(f"../data/{robot}_{mode}_gait_mimickit.pkl")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(motion, f)

    print(f"✅ Saved {record_cycles} cycles MimicKit motion: {save_path}")
    print(f"   frames: {frames.shape}, fps={FPS}")
    print("   One gait cycle duration:", T_cycle)
    
if __name__ == "__main__":
    main()