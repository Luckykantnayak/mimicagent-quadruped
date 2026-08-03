"""
simulator/tools/pose_editor.py

INTERACTIVE pose editor for authoring a robot's named home/rearing STANCES
(the `home_poses:` catalog in robots/<name>/robot.yaml). Open a live MuJoCo
viewer, tilt the base and dial the joints until the robot stands the way you
want on a chosen leg pair, then export a paste-ready YAML entry.

This is the hand-authoring companion to gen_home_poses.py (which SOLVES the five
canonical stances automatically): use this when you want to CHOOSE a pose by eye —
a specific rear angle, a custom tuck, a new stance the auto-solver doesn't cover.
It never changes the runtime; a pose is just prompt DATA (base orientation + stance
feet + per-joint angles) a skill drives through the existing base class.

Runnable from ANY directory (it chdir's to the repo root):

    python simulator/tools/pose_editor.py --robot unitree_go2
    python simulator/tools/pose_editor.py --robot unitree_go2 --load rear --name rear2
    python simulator/tools/pose_editor.py --robot unitree_go2 --dump rear      # headless: print a pose's YAML

Controls (focus the GL window; every action also prints a status line):
    [ ]      select previous / next DOF  (base roll, pitch, yaw, height, then each joint)
    - =      decrease / increase the selected DOF by the current step
    , .      halve / double the step size
    1..9     toggle end-effector N as a STANCE (load-bearing) foot
    A        IK-solve the declared stance feet straight down onto the floor
    G        ground-snap: drop/raise the base so the lowest stance sole rests on z=0
    C        check + print foot heights, penetration, joint-limit status
    P        print the current pose as a YAML home_poses entry
    S        save that YAML entry to --out (appends; defaults to a scratch file)
    R        reset to the model's home keyframe (or the --load pose)
    H        reprint this help
    Q / Esc  quit
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = Path(__file__).resolve().parents[1]
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
os.chdir(REPO_ROOT)                       # data paths (robots/, assets/) are repo-root-relative

from utils.robot_config import load_robot_spec
from utils.scene_compose import compose_model
from utils.math_utils import euler_to_quat
from utils.kinematics import MultiLinkGradientDescentIK

# GLFW key codes are the unshifted-ASCII of printable keys.
K = {c: ord(c) for c in "[]-=,.ACGHPRSQ0123456789"}
K_ESC = 256


def build(robot):
    spec = load_robot_spec(robot, robots_dir="robots")
    if getattr(spec, "kind", "generic") != "quadruped":
        print(f"⚠️  {robot} kind={spec.kind}: rearing stances are a quadruped concept; proceeding anyway.")
    model = compose_model(spec.mjcf, "assets/scenes/flat.xml")
    data = mujoco.MjData(model)
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kf >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kf)
    mujoco.mj_forward(model, data)
    ik = MultiLinkGradientDescentIK(model, data, spec.ee_bodies, sites=spec.ee_sites,
                                    joint_names=spec.actuated_joint_names,
                                    null_space_gain=spec.null_space_gain)
    robot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] > 0]
    home_q7 = data.qpos[7:].copy()
    return spec, model, data, ik, robot_geoms, home_q7, kf


def min_geom_z(model, data, robot_geoms):
    return min(float(data.geom_xpos[g][2] - model.geom_size[g].max()) for g in robot_geoms)


class Pose:
    """Mutable pose state + the DOF cursor the keys drive."""
    def __init__(self, spec, base_z, q7, name, desc):
        self.spec = spec
        self.rpy = [0.0, 0.0, 0.0]
        self.z = float(base_z)
        self.q = np.asarray(q7, dtype=float).copy()          # all joints (incl. passive)
        self.stance = set()                                  # ee names
        self.name = name
        self.desc = desc
        self.step = 0.05
        self.sel = 1                                          # start on base pitch
        # controllable DOFs: 3 base angles + base height + each ACTUATED joint
        self.jidx = [k for k, j in enumerate(spec.joints) if not j.get("passive")]
        self.dof_names = ["base_roll", "base_pitch", "base_yaw", "base_height"] \
            + [spec.joints[k]["name"] for k in self.jidx]

    def value(self, i):
        if i < 3:
            return self.rpy[i]
        if i == 3:
            return self.z
        return float(self.q[self.jidx[i - 4]])

    def nudge(self, delta):
        i = self.sel
        if i < 3:
            self.rpy[i] += delta
        elif i == 3:
            self.z = max(0.03, self.z + delta)
        else:
            k = self.jidx[i - 4]
            lim = self.spec.joints[k].get("limits")
            v = self.q[k] + delta
            if lim:
                v = float(np.clip(v, lim[0], lim[1]))
            self.q[k] = v

    def apply(self, model, data):
        data.qpos[:3] = [0.0, 0.0, self.z]
        data.qpos[3:7] = euler_to_quat(*self.rpy)
        data.qpos[7:] = self.q
        mujoco.mj_forward(model, data)


def offsets(spec):
    return {n: float(o) for n, o in zip(spec.ee_names, spec.ee_ground_offsets)}


def solve_stance(spec, model, data, ik, pose):
    """IK the declared stance feet straight down onto the floor; keep the base fixed."""
    if not pose.stance:
        print("  (no stance feet declared — press 1..%d to pick a leg pair first)" % len(spec.ee_names))
        return
    pose.apply(model, data)
    offs = offsets(spec)
    targets = ik.get_ee_positions().copy()
    mask = np.zeros(len(spec.ee_names), dtype=bool)
    for n in pose.stance:
        i = spec.ee_names.index(n)
        t = targets[i].copy(); t[2] = offs[n]                # sole onto z=0
        targets[i] = t; mask[i] = True
    ik.calculate(targets, active_mask=mask, max_iter=150)
    for k, j in enumerate(spec.joints):                      # clamp to limits
        lim = j.get("limits")
        if lim and not j.get("passive"):
            data.qpos[7 + k] = float(np.clip(data.qpos[7 + k], lim[0], lim[1]))
    mujoco.mj_forward(model, data)
    pose.q = data.qpos[7:].copy()
    print(f"  IK-solved stance {sorted(pose.stance)} onto the floor.")


def ground_snap(spec, model, data, ik, pose):
    pose.apply(model, data)
    offs = offsets(spec)
    ach = ik.get_ee_positions()
    if pose.stance:
        lift = min(ach[spec.ee_names.index(n)][2] - offs[n] for n in pose.stance)
    else:
        lift = min_geom_z(model, data, [g for g in range(model.ngeom) if model.geom_bodyid[g] > 0])
    pose.z -= lift
    print(f"  ground-snapped base z by {-lift:+.3f} -> {pose.z:.3f} m")


def report(spec, model, data, ik, robot_geoms, pose):
    pose.apply(model, data)
    offs = offsets(spec)
    ach = ik.get_ee_positions()
    stance = sorted(pose.stance)
    free = [n for n in spec.ee_names if n not in pose.stance]
    sz = {n: round(float(ach[spec.ee_names.index(n)][2] - offs[n]), 3) for n in stance}
    fz = {n: round(float(ach[spec.ee_names.index(n)][2] - offs[n]), 3) for n in free}
    minz = round(min_geom_z(model, data, robot_geoms), 3)
    in_lim = all((not j.get("passive")) is False or not j.get("limits")
                 or (j["limits"][0] - 1e-3 <= data.qpos[7 + k] <= j["limits"][1] + 1e-3)
                 for k, j in enumerate(spec.joints))
    free_min = min(fz.values()) if fz else 0.0
    verified = bool(minz > -0.02 and in_lim and (not free or free_min > 0.10))
    print(f"  stance sole-z {sz}  | free lift {fz}  | min_geom_z {minz}"
          f"  | in_limits {in_lim}  | VERIFIED {verified}")
    return verified


def yaml_entry(spec, pose, verified):
    jnames = [j["name"] for j in spec.joints]
    jdict = {jnames[k]: round(float(pose.q[k]), 3)
             for k in range(len(jnames)) if not spec.joints[k].get("passive")}
    stance = sorted(pose.stance) if pose.stance else list(spec.ee_names)
    lines = [
        f"  - name: {pose.name}",
        f'    description: "{pose.desc}"',
        f"    stance_ees: [{', '.join(stance)}]",
        f"    base_rpy: [{pose.rpy[0]:.3f}, {pose.rpy[1]:.3f}, {pose.rpy[2]:.3f}]",
        f"    base_height: {pose.z:.3f}",
        f"    verified: {verified}",
        "    joints: {" + ", ".join(f"{k}: {v}" for k, v in jdict.items()) + "}",
    ]
    return "\n".join(lines)


def render_views(model, data, prefix, azimuths=(125, 200), w=760, h=560):
    """Render the CURRENT data pose from a few free-camera angles to PNGs. Uses the
    glfw GL backend (the one that works offscreen here). Returns the file paths."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    try:
        import imageio.v2 as imageio
        save = lambda p, im: imageio.imwrite(p, im)
    except Exception:
        from PIL import Image
        save = lambda p, im: Image.fromarray(im).save(p)
    cam = mujoco.MjvCamera()
    # frame on the robot's bounding box so it stays fully in view at any pose height
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] > 0]
    pts = np.array([data.geom_xpos[g] for g in gids])
    center = (pts.max(0) + pts.min(0)) / 2.0
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    cam.lookat[:] = center
    cam.distance = max(1.6, 1.7 * diag)
    cam.elevation = -12
    r = mujoco.Renderer(model, h, w)
    outs = []
    for i, az in enumerate(azimuths):
        cam.azimuth = az
        r.update_scene(data, camera=cam)
        p = f"{prefix}_{i+1}.png"
        save(p, r.render())
        outs.append(p)
    return outs


HELP = __doc__.split("Controls", 1)[1]


def main():
    ap = argparse.ArgumentParser(description="Interactive home/rearing pose editor.")
    ap.add_argument("--robot", required=True)
    ap.add_argument("--load", help="start from an existing home_poses entry (by name)")
    ap.add_argument("--name", default="custom_pose", help="name for the pose you author")
    ap.add_argument("--out", help="file to append saved YAML entries to")
    ap.add_argument("--dump", help="headless: load a home_poses entry, print its report + YAML, exit")
    ap.add_argument("--shot", help="headless: apply a pose from --pose JSON, render angles to <SHOT>_N.png, print report+YAML")
    ap.add_argument("--pose", help="JSON file with the working pose (for --shot / --live)")
    ap.add_argument("--live", action="store_true",
                    help="open a live viewer that RELOADS --pose JSON whenever it changes "
                         "(so it can be driven externally while you watch)")
    args = ap.parse_args()

    spec, model, data, ik, robot_geoms, home_q7, kf = build(args.robot)

    def load_named(name):
        for hp in getattr(spec, "home_poses", []):
            if hp.get("name") == name:
                return hp
        raise SystemExit(f"no home_poses entry named '{name}' in robots/{args.robot}/robot.yaml")

    def pose_from(hp, name, desc):
        q = home_q7.copy()
        jn = [j["name"] for j in spec.joints]
        for k, v in (hp.get("joints") or {}).items():
            if k in jn:
                q[jn.index(k)] = float(v)
        p = Pose(spec, hp.get("base_height", spec.base_height_range[0]), q, name, desc)
        p.rpy = list(hp.get("base_rpy", [0.0, 0.0, 0.0]))
        p.stance = set(hp.get("stance_ees", []))
        return p

    # headless dump path (no display needed) — round-trips a stored pose through the same math
    if args.dump:
        hp = load_named(args.dump)
        p = pose_from(hp, hp["name"], hp.get("description", ""))
        v = report(spec, model, data, ik, robot_geoms, p)
        print(yaml_entry(spec, p, v))
        return

    if args.shot:
        import json
        pj = json.load(open(args.pose)) if args.pose and os.path.isfile(args.pose) else {}
        if pj.get("base_height") is None:
            pj["base_height"] = float(data.qpos[2])          # default to home base height
        name = pj.get("name", args.name)
        desc = pj.get("description", f"custom pose {name}")
        p = pose_from(pj, name, desc)
        if pj.get("ik_stance"):
            solve_stance(spec, model, data, ik, p)
        if pj.get("ground_snap"):
            ground_snap(spec, model, data, ik, p)
        v = report(spec, model, data, ik, robot_geoms, p)
        outs = render_views(model, data, args.shot)
        print("RENDERED: " + " ".join(outs))
        print(yaml_entry(spec, p, v))
        resolved = dict(name=name, description=desc,
                        base_rpy=[round(x, 3) for x in p.rpy], base_height=round(p.z, 3),
                        stance_ees=sorted(p.stance),
                        joints={spec.joints[k]["name"]: round(float(p.q[k]), 3)
                                for k in range(len(spec.joints)) if not spec.joints[k].get("passive")},
                        ik_stance=False, ground_snap=False)
        print("RESOLVED_JSON=" + json.dumps(resolved))
        return

    if args.live:
        import json, time
        try:
            import mujoco.viewer
        except Exception as e:                              # pragma: no cover
            raise SystemExit(f"mujoco.viewer unavailable ({e}); use --shot for rendered stills.")
        home_base_z = float(data.qpos[2])

        def load_and_apply():
            pj = json.load(open(args.pose))
            if pj.get("base_height") is None:
                pj["base_height"] = home_base_z
            p = pose_from(pj, pj.get("name", "custom_pose"), pj.get("description", ""))
            if pj.get("ik_stance"):
                solve_stance(spec, model, data, ik, p)
            if pj.get("ground_snap"):
                ground_snap(spec, model, data, ik, p)
            report(spec, model, data, ik, robot_geoms, p)

        print(f"LIVE viewer up — polling {args.pose}; edit that JSON to update the pose.")
        last_m = None
        bbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        with mujoco.viewer.launch_passive(model, data, key_callback=lambda k: None) as viewer:
            # camera TRACKS the base body at a fixed relative distance, so the robot
            # stays framed and the same size whatever pose/height it takes.
            if bbid >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = bbid
                viewer.cam.distance = 2.2
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -12
            while viewer.is_running():
                try:
                    m = os.path.getmtime(args.pose) if os.path.isfile(args.pose) else None
                    if m != last_m:
                        last_m = m
                        try:
                            load_and_apply()
                        except Exception as e:
                            print(f"  (skip malformed/partial pose: {e})")
                except Exception:
                    pass
                viewer.sync()
                time.sleep(0.15)
        return

    if args.load:
        hp = load_named(args.load)
        pose = pose_from(hp, args.name, hp.get("description", f"custom pose {args.name}"))
    else:
        pose = Pose(spec, float(data.qpos[2]), home_q7, args.name, f"custom pose {args.name}")

    try:
        import mujoco.viewer
    except Exception as e:                                   # pragma: no cover
        raise SystemExit(f"mujoco.viewer unavailable ({e}); use --dump for headless, or run with a display.")

    pending = []

    def on_key(keycode):
        pending.append(keycode)

    def status():
        dof = pose.dof_names[pose.sel]
        print(f"  ▶ {dof} = {pose.value(pose.sel):+.3f} | step {pose.step:.3f} | "
              f"stance {sorted(pose.stance) or '[]'} | base_rpy "
              f"[{pose.rpy[0]:+.2f},{pose.rpy[1]:+.2f},{pose.rpy[2]:+.2f}] z {pose.z:.3f}")

    print("=" * 78)
    print(f"POSE EDITOR — {args.robot} — authoring '{pose.name}'")
    print("Controls" + HELP)
    print("EE index map:  " + "  ".join(f"{i+1}={n}" for i, n in enumerate(spec.ee_names)))
    print("=" * 78)
    status()

    outfile = args.out or str(REPO_ROOT / "skill_output" / f"{args.robot}_poses.yaml")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        pose.apply(model, data)
        viewer.sync()
        while viewer.is_running():
            while pending:
                key = pending.pop(0)
                if key in (K_ESC, K["Q"]):
                    viewer.close(); break
                elif key == K["["]:
                    pose.sel = (pose.sel - 1) % len(pose.dof_names); status()
                elif key == K["]"]:
                    pose.sel = (pose.sel + 1) % len(pose.dof_names); status()
                elif key == K["-"]:
                    pose.nudge(-pose.step); pose.apply(model, data); status()
                elif key == K["="]:
                    pose.nudge(+pose.step); pose.apply(model, data); status()
                elif key == K[","]:
                    pose.step = max(0.005, pose.step / 2); status()
                elif key == K["."]:
                    pose.step = min(1.0, pose.step * 2); status()
                elif ord("1") <= key <= ord("9"):
                    i = key - ord("1")
                    if i < len(spec.ee_names):
                        n = spec.ee_names[i]
                        pose.stance ^= {n}
                        print(f"  stance {sorted(pose.stance)}")
                elif key == K["A"]:
                    solve_stance(spec, model, data, ik, pose); pose.apply(model, data); status()
                elif key == K["G"]:
                    ground_snap(spec, model, data, ik, pose); pose.apply(model, data); status()
                elif key == K["C"]:
                    report(spec, model, data, ik, robot_geoms, pose)
                elif key == K["P"]:
                    v = report(spec, model, data, ik, robot_geoms, pose)
                    print("\n" + yaml_entry(spec, pose, v) + "\n")
                elif key == K["S"]:
                    v = report(spec, model, data, ik, robot_geoms, pose)
                    os.makedirs(os.path.dirname(outfile), exist_ok=True)
                    with open(outfile, "a", encoding="utf-8") as f:
                        f.write(yaml_entry(spec, pose, v) + "\n")
                    print(f"  saved '{pose.name}' -> {outfile}")
                elif key == K["R"]:
                    pose.rpy = [0.0, 0.0, 0.0]; pose.z = float(home_q7[0]) * 0 + float(data.qpos[2])
                    if kf >= 0:
                        mujoco.mj_resetDataKeyframe(model, data, kf); mujoco.mj_forward(model, data)
                    pose.z = float(data.qpos[2]); pose.q = home_q7.copy(); pose.stance.clear()
                    print("  reset to home keyframe"); status()
                elif key == K["H"]:
                    print("Controls" + HELP)
                    print("EE index map:  " + "  ".join(f"{i+1}={n}" for i, n in enumerate(spec.ee_names)))
            viewer.sync()

    print(f"\nDone. Saved poses (if any) are in {outfile} — paste them under `home_poses:` in "
          f"robots/{args.robot}/robot.yaml.")


if __name__ == "__main__":
    main()
