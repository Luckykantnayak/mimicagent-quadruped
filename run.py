# core/app.py
from core.config import load_config
from core.simulator import MujocoSimulator
from core.controller import MotionControllerRunner
from core.recorder import MimicKitRecorder
from pathlib import Path
import glfw

import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    # --------------------------
    # Load configuration
    # --------------------------
    args = load_config()

    # --------------------------
    # Initialize simulator
    # --------------------------
    scene_path = f"./assets/unitree_{args.robot}/scene.xml"
    sim = MujocoSimulator(scene_path, init_position=args.init_position, sim_freq=int(args.sim_freq))

    # --------------------------
    # Initialize gait controller
    # --------------------------
    runner = MotionControllerRunner(sim, args)

    # --------------------------
    # Initialize recorder (optional)
    # --------------------------
    recorder = MimicKitRecorder(fps=int(args.sim_freq)) if args.record else None

    # --------------------------
    # Compute total frames to save 
    # --------------------------
    total_frames = int(args.record_cycles / args.gait_freq * args.sim_freq)
    start_frame = 50 
    end_frame = start_frame + total_frames
    # --------------------------
    # Main loop
    # --------------------------
    while not glfw.window_should_close(sim.window):
        # Step controller
        state = runner.step()

        # Record if needed
        if recorder :
            if runner.frame >= start_frame and runner.frame < end_frame:
               recorder.record(state)
            elif runner.frame == end_frame:
                print(f"Recorded {total_frames} Frames")
                break

        # Render simulation
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
            out_path = Path(f"./data/{args.version}/{args.robot}_{getattr(args, 'skating_style', 'default')}_{args.mode}_gait_mimickit.pkl")
        else:
            out_path = Path(f"./data/{args.version}/{args.mode}.pkl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        recorder.save(out_path)

if __name__ == "__main__":
    main()
