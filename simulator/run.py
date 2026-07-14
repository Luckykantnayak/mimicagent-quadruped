# run.py
from core.config import load_config
from core.simulator import MujocoSimulator
from core.controller import MotionControllerRunner
from core.recorder import MimicKitRecorder
from utils.robot_config import load_robot_spec

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
    # Load robot spec → scene path, camera, end-effectors
    # --------------------------
    robots_dir = getattr(args, "robots_dir", "robots")
    spec = load_robot_spec(args.robot, robots_dir=robots_dir)

    print(f"[run] Robot   : {spec.name}")
    print(f"[run] Scene   : {spec.scene_xml}")
    print(f"[run] Mode    : {args.mode}")

    # --------------------------
    # Initialize simulator
    # --------------------------
    sim = MujocoSimulator(
        spec.scene_xml,
        init_position=args.init_position,
        sim_freq=int(args.sim_freq),
        base_body=spec.base_body,
        camera=spec.camera,
    )

    # --------------------------
    # Initialize gait controller
    # --------------------------
    runner = MotionControllerRunner(sim, args)

    # --------------------------
    # Initialize recorder (optional)
    # --------------------------
    recorder = (
        MimicKitRecorder(fps=int(args.sim_freq), robot=spec.name, ee_names=spec.ee_names)
        if args.record
        else None
    )

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
        out_path = Path(f"./skill_output/{args.robot}/{args.mode}/{args.mode}.pkl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        recorder.save(out_path)
        print(f"[run] Saved → {out_path}")


if __name__ == "__main__":
    main()
