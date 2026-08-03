"""
simulator/tools/record_video.py

Offscreen video recorder: steps a skill through N phase cycles and renders each
frame with mujoco.Renderer (no display needed), then encodes an mp4. Reuses the
same MotionControllerRunner as the live sim, so what it records is exactly what
plays on screen (grounding, object mocap, IK — all identical).

Run from the repo root:
  python simulator/tools/record_video.py --robot humanoid --mode SKILL_CARRYBOX \
      --scene assets/scenes/carry.xml --grounding off \
      --out /tmp/carrybox.mp4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco
import imageio

PROJECT_ROOT = Path(__file__).resolve().parents[1]      # simulator/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.simulator import MujocoSimulator
from core.controller import MotionControllerRunner
from utils.robot_config import load_robot_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--grounding", default="off")
    ap.add_argument("--robots_dir", default="robots")
    ap.add_argument("--sim_freq", type=float, default=200)
    ap.add_argument("--init_position", default="0 0 0")
    ap.add_argument("--cycles", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cam_lookat", default=None, help="x,y,z")
    ap.add_argument("--cam_dist", type=float, default=None)
    ap.add_argument("--cam_az", type=float, default=None)
    ap.add_argument("--cam_el", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = load_robot_spec(args.robot, robots_dir=args.robots_dir)
    scene = args.scene or spec.scene_xml
    sim = MujocoSimulator(scene, init_position=args.init_position,
                          sim_freq=int(args.sim_freq), base_body=spec.base_body,
                          camera=spec.camera, headless=True,
                          robot_mjcf=spec.mjcf)   # compose robot + agnostic scene at load
    runner = MotionControllerRunner(sim, args)
    model, data = sim.model, sim.data

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    c = spec.camera
    lk = ([float(v) for v in args.cam_lookat.split(",")] if args.cam_lookat
          else c.get("lookat", [0.0, 0.0, 0.9]))
    cam.lookat[:] = lk
    cam.distance = args.cam_dist if args.cam_dist is not None else c.get("distance", 4.0)
    cam.azimuth = args.cam_az if args.cam_az is not None else c.get("azimuth", 90)
    cam.elevation = args.cam_el if args.cam_el is not None else c.get("elevation", -15)

    skill_freq = getattr(runner.motion_gen, "freq", None) or 1.0
    steps = int(round(args.cycles / skill_freq * args.sim_freq))
    stride = max(1, int(round(args.sim_freq / args.fps)))

    frames = []
    for i in range(steps):
        runner.step()
        if i % stride == 0:
            renderer.update_scene(data, cam)
            frames.append(renderer.render())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if str(args.out).lower().endswith(".gif"):
        # GIF: play FORWARD on an infinite loop (loop=0), never ping-pong/backward —
        # frames are appended in time order, so this always reads forward.
        imageio.mimwrite(args.out, frames, fps=args.fps, loop=0)
    else:
        imageio.mimwrite(args.out, frames, fps=args.fps, quality=8, macro_block_size=None)
    print(f"[record_video] wrote {len(frames)} frames ({len(frames)/args.fps:.1f}s) -> {args.out}")


if __name__ == "__main__":
    main()
