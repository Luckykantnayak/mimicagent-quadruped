# run.py
from core.config import load_config
from core.simulator import MujocoSimulator
from core.controller import MotionControllerRunner
from core.recorder import MimicKitRecorder
from utils.robot_config import load_robot_config

from pathlib import Path
import glfw
import sys

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    # --------------------------
    # Load configuration
    # --------------------------
    args = load_config()

    # --------------------------
    # Load robot config → get scene path
    # --------------------------
    robots_dir = getattr(args, "robots_dir", "robots")
    robot_cfg = load_robot_config(args.robot, robots_dir=robots_dir)
    scene_path = robot_cfg.scene_xml

    print(f"[run] Robot   : {robot_cfg.name}")
    print(f"[run] Scene   : {scene_path}")
    print(f"[run] Mode    : {args.mode}")

    # --------------------------
    # Load robot config early
    # --------------------------
    robots_dir = getattr(args, "robots_dir", "robots")
    robot_cfg = load_robot_config(args.robot, robots_dir=robots_dir)

    # --------------------------
    # Initialize simulator
    # --------------------------
    sim = MujocoSimulator(
        scene_path,
        init_position=args.init_position,
        sim_freq=int(args.sim_freq),
        base_body=robot_cfg.base_body,  # ← "pelvis" for humanoid, "base" for go2
    )

    # --------------------------
    # Initialize gait controller
    # --------------------------
    runner = MotionControllerRunner(sim, args)

    # --------------------------
    # Initialize recorder (optional)
    # --------------------------
    recorder = MimicKitRecorder(fps=int(args.sim_freq)) if args.record else None

    # --------------------------
    # Compute total frames to record
    # --------------------------
    total_frames = int(args.record_cycles / args.gait_freq * args.sim_freq)
    start_frame = 50
    end_frame = start_frame + total_frames

    # --------------------------
    # Main loop
    # --------------------------
    while not glfw.window_should_close(sim.window):
        state = runner.step()

        if recorder:
            if runner.frame >= start_frame and runner.frame < end_frame:
                recorder.record(state)
            elif runner.frame == end_frame:
                print(f"Recorded {total_frames} frames")
                break

        sim.render()

    # --------------------------
    # Close simulation
    # --------------------------
    sim.close()

    # --------------------------
    # Save recorded motion
    # --------------------------
    if recorder:
        if args.mode == "skating":
            out_path = Path(
                f"./data/{args.version}/"
                f"{args.robot}_{getattr(args, 'skating_style', 'default')}_{args.mode}_gait_mimickit.pkl"
            )
        else:
            out_path = Path(f"./data/{args.version}/{args.robot}_{args.mode}.pkl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        recorder.save(out_path)
        print(f"[run] Saved → {out_path}")


if __name__ == "__main__":
    main()