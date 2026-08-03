"""
Offline re-scorer for every skill already in skill_output/.

WHY THIS EXISTS: `finalize` ships a skill and DELETES its per-iteration
`report_*.json`, so after a run there is no record of how the shipped motion
actually scored — and therefore no way to answer "did that framework change help
or hurt?". This re-runs the SAME evaluator the pipeline's unit-test node runs
(`utils.test_motion_quality.evaluate_trajectory`) over the shipped trajectories,
prints a regression table, and can snapshot to JSON and diff two snapshots.

It is READ-ONLY with respect to skill_output: it loads `<skill>.pkl` and
`<skill>_plan.json`, and writes nothing but the snapshot you ask for.

Usage (from the repo root or from agent/):

    python agent/score_skills.py                          # table for the configured robot
    python agent/score_skills.py --json before.json       # ... and snapshot it
    python agent/score_skills.py --compare before.json    # score now, diff vs the snapshot
    python agent/score_skills.py --skill SKILL_CLIMB      # just one
"""

import argparse
import json
import os
import pickle as pkl
import sys

# Runnable from the repo root as well as from agent/ (the package layout assumes
# agent/ is on sys.path — agent/utils deliberately shadows the simulator's utils).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts.prompt_loader import load_yaml_config                # noqa: E402
from utils.plan_io import load_plan_dict, extract_invariants, plan_meta  # noqa: E402
from utils.robot_spec import load_robot_spec                      # noqa: E402
from utils.test_motion_quality import evaluate_trajectory         # noqa: E402

# The metric that best expresses "how badly" each check failed, for the table and
# the diff. Mirrors _SEVERITY_SCALE in agents/code_gen/run_unit_test.py.
_KEY_METRIC = {
    "surface_penetration": "max_penetration",
    "self_collision": "max_overlap",
    "contact_no_slide": "max_stance_drift",
    "contact_grounded": "float_clearance",
    "no_teleport": "worst_step",
    "ee_no_teleport": "worst_step",
    "joint_velocity_envelope": "worst_ratio",
    "framework_interventions": "clamp_max",
    "support_or_ballistic": "rest_frac",
    "clip_duration": "clip_s",
}


def score_skill(skill_dir, skill_name, spec):
    """Evaluate one shipped skill. Returns a result dict, or None if unscoreable."""
    traj_path = os.path.join(skill_dir, f"{skill_name}.pkl")
    if not os.path.isfile(traj_path):
        return None
    with open(traj_path, "rb") as f:
        traj = pkl.load(f)

    plan = load_plan_dict(skill_dir, skill_name)
    invariants, sliding, pivots = extract_invariants(plan)
    env, cyclic, _reps = plan_meta(plan)

    fb = evaluate_trajectory(
        traj=traj, spec=spec, dt=1.0 / traj["fps"],
        invariants=invariants, sliding_contacts=sliding,
        pivot_contacts=pivots, cyclic=cyclic,
    )
    summary = fb["summary"]

    failures = {}
    for t in fb["tests"]:
        if t["category"] != "discrete" or t["passed"] or t.get("skipped"):
            continue
        key = _KEY_METRIC.get(t["name"])
        val = t.get("metrics", {}).get(key) if key else None
        failures[t["name"]] = round(float(val), 4) if isinstance(val, (int, float)) else None

    # Resolve through the spec even when the plan named no environment: an absent
    # `environment` means the implicit `flat` default, which has a real grounding
    # mode — reporting None there would hide the very setting we are auditing.
    try:
        resolved = spec.get_environment(env) if env else spec.all_environments[0]
        g = resolved.get("grounding")
        grounding = "off" if g is False else ("contact" if g is True else g)
    except Exception:
        grounding = None

    return dict(
        skill=skill_name,
        environment=env or "flat",
        grounding=grounding,
        floor_pass=bool(summary["passed_all_discrete"]),
        reward=round(float(summary["total_reward"]), 2),
        fidelity_passed=summary.get("num_fidelity_passed", 0),
        fidelity_total=summary.get("num_fidelity_total", 0),
        fidelity_failed=summary.get("fidelity_failed", []),
        failures=failures,
        frames=int(len(traj["root_pos"])),
        fps=float(traj["fps"]),
    )


def collect(cfg, spec, only=None):
    root = os.path.join(cfg["paths"]["skill_output_dir"], spec.name)
    if not os.path.isdir(root):
        print(f"No skill output for robot '{spec.name}' at {root}")
        return []
    results = []
    for name in sorted(os.listdir(root)):
        if only and name != only:
            continue
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        r = score_skill(d, name, spec)
        if r:
            results.append(r)
    return results


def print_table(results):
    if not results:
        print("(nothing to score)")
        return
    w = max(len(r["skill"]) for r in results)
    n_pass = sum(1 for r in results if r["floor_pass"])
    print(f"\n{'skill'.ljust(w)}  {'env':<10} {'ground':<8} {'floor':<6} "
          f"{'reward':>7}  fidelity")
    print("-" * (w + 46))
    for r in results:
        fid = f"{r['fidelity_passed']}/{r['fidelity_total']}"
        print(f"{r['skill'].ljust(w)}  {r['environment']:<10} {str(r['grounding']):<8} "
              f"{'PASS' if r['floor_pass'] else 'FAIL':<6} {r['reward']:>7.2f}  {fid}")
        for name, val in r["failures"].items():
            shown = f" = {val}" if val is not None else ""
            print(f"{' ' * w}    ✗ {name}{shown}")
    print("-" * (w + 46))
    print(f"floor: {n_pass}/{len(results)} passing\n")


def print_diff(before, after):
    """Diff two runs. The point is to make a framework change's effect legible:
    which skills flipped, and which individual checks got better or worse."""
    b = {r["skill"]: r for r in before}
    a = {r["skill"]: r for r in after}
    print("\n=== DIFF (baseline -> now) ===")
    for name in sorted(set(b) | set(a)):
        rb, ra = b.get(name), a.get(name)
        if rb is None:
            print(f"  + {name}: NEW ({'PASS' if ra['floor_pass'] else 'FAIL'})")
            continue
        if ra is None:
            print(f"  - {name}: GONE")
            continue
        bits = []
        if rb["floor_pass"] != ra["floor_pass"]:
            bits.append("FLOOR FIXED ✅" if ra["floor_pass"] else "FLOOR BROKEN ❌")
        for chk in sorted(set(rb["failures"]) | set(ra["failures"])):
            vb, va = rb["failures"].get(chk, "—"), ra["failures"].get(chk, "—")
            if chk not in ra["failures"]:
                bits.append(f"fixed {chk} (was {vb})")
            elif chk not in rb["failures"]:
                bits.append(f"BROKE {chk} (now {va})")
            elif isinstance(vb, float) and isinstance(va, float) and abs(va - vb) > 1e-4:
                bits.append(f"{chk} {vb}->{va}{' better' if va < vb else ' WORSE'}")
        if rb["fidelity_passed"] != ra["fidelity_passed"]:
            bits.append(f"fidelity {rb['fidelity_passed']}->{ra['fidelity_passed']}"
                        f"/{ra['fidelity_total']}")
        print(f"  = {name}: {'; '.join(bits) if bits else 'unchanged'}")
    nb = sum(1 for r in before if r["floor_pass"])
    na = sum(1 for r in after if r["floor_pass"])
    print(f"\nfloor passing: {nb}/{len(before)} -> {na}/{len(after)}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=None, help="override cfg robot.name")
    ap.add_argument("--skill", default=None, help="score only this skill")
    ap.add_argument("--json", default=None, help="write a snapshot here")
    ap.add_argument("--compare", default=None, help="diff against a previous snapshot")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = load_yaml_config(os.path.join(here, "cfg", "config.yaml"))
    if args.robot:
        cfg["robot"]["name"] = args.robot
    spec = load_robot_spec(cfg)

    results = collect(cfg, spec, only=args.skill)
    print_table(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"snapshot -> {args.json}")

    if args.compare:
        with open(args.compare) as f:
            print_diff(json.load(f), results)

    # Non-zero exit when anything fails its floor, so this can gate a script.
    return 0 if all(r["floor_pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
