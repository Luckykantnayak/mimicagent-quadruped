# run.py
from core.config import load_config
from core.simulator import MujocoSimulator
from core.controller import MotionControllerRunner
from core.recorder import MimicKitRecorder
from utils.robot_config import load_robot_spec
from utils.plan_io import load_plan_dict, plan_meta
from motion import has_skill_version

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
    # Resolve HOW this skill was recorded
    # --------------------------
    # Precedence for scene / grounding / record_cycles:
    #     explicit CLI flag  >  the skill's own environment  >  the config default
    # A skill's environment comes from its curated exemplar entry, or — for a
    # GENERATED skill — from the motion plan the agent wrote beside it. Resolving
    # it here is what makes `python simulator/run.py --mode <SKILL> --record True`
    # reproduce the agent's recording instead of quietly differing from it (wrong
    # grounding floats the contacts; wrong cycle count wraps a one-shot skill back
    # to its start pose mid-clip and reads as a gross teleport).
    scene_xml = args.scene
    grounding = args.grounding
    record_cycles = args.record_cycles

    # Resolution order MUST match the controller's skill-loading order, or we would
    # configure the run for one skill and then play another: the controller prefers a
    # GENERATED skill over a same-named curated exemplar, so a name registered as both
    # (SKILL_CARRYBOX is) must resolve from the generated skill's plan, not the
    # exemplar's environment.
    env, source = None, None
    gen_root = Path("skill_output") / spec.name
    skill_dir = gen_root / args.mode
    plan = load_plan_dict(str(skill_dir), args.mode) if has_skill_version(
        str(gen_root), args.mode) else {}
    if plan:
        env_name, _cyclic, reps = plan_meta(plan)
        if record_cycles is None:
            record_cycles = reps
        if env_name:
            env, source = spec.get_environment(env_name), "plan"
    if env is None:
        ex = spec.get_exemplar(args.mode)
        if ex:
            env, source = spec.get_environment(ex.get("environment")), "exemplar"
            if record_cycles is None and ex.get("repetitions") is not None:
                record_cycles = int(ex["repetitions"])

    if env:
        if scene_xml is None:
            scene_xml = env.get("scene")
        if grounding is None:
            grounding = env.get("grounding")
        print(f"[run] {source}: '{args.mode}' -> environment '{env.get('name')}'")

    # YAML parses an unquoted `off`/`on` as a bool — normalize back to a mode name.
    if isinstance(grounding, bool):
        grounding = "off" if grounding is False else "contact"
    args.grounding = grounding if grounding is not None else args.cfg_grounding
    args.record_cycles = record_cycles if record_cycles is not None else args.cfg_record_cycles

    # --------------------------
    # Initialize simulator
    # --------------------------
    scene_xml = scene_xml or spec.scene_xml
    print(f"[run] Scene xml: {scene_xml}")
    print(f"[run] Grounding: {args.grounding} | record_cycles: {args.record_cycles}")
    sim = MujocoSimulator(
        scene_xml,
        init_position=args.init_position,
        sim_freq=int(args.sim_freq),
        base_body=spec.base_body,
        camera=spec.camera,
        headless=getattr(args, "headless", False),
        robot_mjcf=spec.mjcf,       # compose robot + agnostic scene at load
    )

    # --------------------------
    # Initialize gait controller
    # --------------------------
    runner = MotionControllerRunner(sim, args)

    # --------------------------
    # Initialize recorder (optional)
    # --------------------------
    recorder = (
        MimicKitRecorder(fps=int(args.sim_freq), robot=spec.name,
                         ee_names=getattr(runner, "contact_names", spec.ee_names),
                         support_ees=runner.support_ees,
                         limb_radius=getattr(runner, "limb_radius", None),
                         object_extents=getattr(runner, "object_extents", None),
                         passive_joint_indices=spec.passive_joint_indices)
        if args.record
        else None
    )

    # --------------------------
    # Compute total frames to record
    # --------------------------
    # Phase advances at the SKILL's own freq (set in its __init__), which may
    # differ from the config gait_freq. Record a whole number of the skill's
    # phase cycles so a slow/long skill (freq well below 1) is captured in FULL
    # instead of a truncated fraction — the unit tests judge the whole motion.
    skill_freq = getattr(runner.motion_gen, "freq", None) or args.gait_freq
    total_frames = int(round(args.record_cycles / skill_freq * args.sim_freq))
    warmup_frames = 50
    # SETTLE WINDOW — after the phase clock is reset to 0, HOLD phase 0 for a short
    # window so IK fully converges from the warmup pose to the EXACT phase-0 pose
    # BEFORE capture. Without it the ACHIEVED contacts drag toward their pins over
    # the first ~0.1 s of the recording (the authored targets never move, but the
    # realized feet catch up), which reads as a phantom contact_no_slide drift on
    # the opening frames. General; independent of skill/morphology.
    settle_frames = int(round(0.3 * args.sim_freq))
    start_frame = warmup_frames + settle_frames
    end_frame = start_frame + total_frames

    # --------------------------
    # Main loop
    # --------------------------
    while not sim.should_close():
        # End of warmup: restart the skill's phase clock at 0 and clear the seam
        # bridge, so the recorded window is EXACTLY one cycle from phase 0 (no loop
        # seam — a NON-cyclic pick-and-place would otherwise snap its box back at
        # the wrap) and frame 0 snaps to the authored pose, not the warmup one.
        if recorder and runner.frame == warmup_frames:
            runner.motion_gen.t = 0.0
            runner.reset_bridge()
        # Settle window: keep the phase pinned at 0 (advance nothing, record nothing)
        # so IK reaches the phase-0 pose before the first captured frame. The bound is
        # INCLUSIVE of start_frame (<=): base.step() reads t THEN increments it, so
        # holding through start_frame makes the FIRST recorded frame land exactly on
        # phase 0 and the LAST on (N-1)/N — endpoint-EXCLUSIVE. Without the <=, the last
        # recorded frame lands on phase 1.0, which (freq*t)%1 WRAPS back to 0, snapping a
        # one-shot skill's final frame to its start pose (fake "0 cm net motion", a
        # phantom stance-foot slide, and a seam teleport — which then poisons the loop).
        if recorder and warmup_frames <= runner.frame <= start_frame:
            runner.motion_gen.t = 0.0

        state = runner.step()

        if recorder:
            if start_frame < runner.frame <= end_frame:
                recorder.record(state)
            elif runner.frame > end_frame:
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
