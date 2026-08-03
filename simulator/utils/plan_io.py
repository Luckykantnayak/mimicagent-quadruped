"""
simulator/utils/plan_io.py

Reading the motion PLAN back off disk — the ONE reader both halves share.

The plan is a shared on-disk contract, not agent-private state: the agent WRITES
it, and the simulator READS it to replay a generated skill exactly as it was
recorded (same scene, same grounding, same cycle count). It lives here, on the
simulator side, because `agent/utils` shadows this package — the agent bridges to
it by file path via `agent/utils/plan_io.py`, the same way it bridges to
`robot_config.py`. Keeping one implementation is the point: a second copy is how
the replayed motion silently stops matching the recorded one.

The planner OWNS the fidelity invariants and the declared sliding/pivot contacts,
so every consumer reads them from here and never from the generated code (the
critic-coder must not be able to edit its own tests).

Two on-disk shapes exist, because `finalize` rewrites the skill directory:
  - DURING a run:   motion_plan_0001.json, motion_plan_0002.json, ...
  - AFTER finalize: <skill>_plan.json      (the per-iteration files are deleted)
`load_plan_dict` accepts either, newest-first.
"""

import json
import os
import re


def load_plan_dict(skill_dir, skill_name=None):
    """The motion plan the code was built from, as a dict ({} if unavailable).

    Tries the per-iteration `motion_plan_NNNN.json` first (a run in progress),
    then the finalized `<skill>_plan.json`. Never raises: a missing or malformed
    plan degrades to {}, which every caller reads as "nothing was declared".
    """
    for path in _candidate_paths(skill_dir, skill_name):
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (OSError, ValueError):
            continue
    return {}


def _candidate_paths(skill_dir, skill_name):
    latest = _latest_numbered(skill_dir, "motion_plan")
    if latest:
        yield latest
    name = skill_name or os.path.basename(os.path.normpath(skill_dir))
    yield os.path.join(skill_dir, f"{name}_plan.json")


def _latest_numbered(dir_path, base):
    """Path to the highest-numbered <base>_NNNN.json in dir_path, or None."""
    if not os.path.isdir(dir_path):
        return None
    pattern = re.compile(rf"^{re.escape(base)}_(\d+)\.json$")
    best, best_n = None, -1
    for fname in os.listdir(dir_path):
        m = pattern.match(fname)
        if m and int(m.group(1)) > best_n:
            best, best_n = fname, int(m.group(1))
    return os.path.join(dir_path, best) if best else None


def _nested(plan):
    """The planner emits its body either top-level or under "motion_plan"."""
    if not isinstance(plan, dict):
        return {}, {}
    mp = plan.get("motion_plan", {}) if isinstance(plan.get("motion_plan"), dict) else {}
    return plan, mp


def _pick(plan, key, default=None):
    top, mp = _nested(plan)
    v = top.get(key)
    if v is None:
        v = mp.get(key)
    return default if v is None else v


def extract_invariants(plan):
    """(fidelity invariants, sliding contacts, pivot contacts) from a plan.

    Sliding contacts (declared `sliding_contacts`) are FULLY exempt from the
    no-slide law; a 'pivot'-mode contact spins on the spot — its EE ORIGIN traces
    a small circle — so it gets a BOUNDED drift budget (handled in
    eval_contact_no_slide), NOT a full exemption, so a real pirouette passes but a
    runaway 'pivot' slide still fails.
    """
    if not isinstance(plan, dict):
        return [], [], []
    top, mp = _nested(plan)

    invariants = _pick(plan, "fidelity_invariants", [])
    if not isinstance(invariants, list):
        invariants = []

    sliding = _pick(plan, "sliding_contacts", [])
    exempt = list(sliding) if isinstance(sliding, list) else []

    pivots = []
    for cs in (mp.get("contact_schedule") or top.get("contact_schedule") or []):
        if not isinstance(cs, dict):
            continue
        for c in (cs.get("contacts") or []):
            if isinstance(c, dict) and c.get("mode") == "pivot" and c.get("ee") \
                    and c["ee"] not in pivots:
                pivots.append(c["ee"])
    return invariants, exempt, pivots


def plan_meta(plan):
    """The replay-relevant scalars: (environment, cyclic, repetitions).

    `repetitions` is what the recorder needs as --record_cycles. Getting it wrong
    is not cosmetic: recording 2 passes of a ONE-SHOT skill (repetitions=1) wraps
    phase 1 -> 0 and snaps the robot back to its start pose mid-clip, which reads
    downstream as a gross joint teleport. Clamped to the same 1..8 band the agent
    uses in code_executer_agent.run_skill.
    """
    env = _pick(plan, "environment")
    cyclic = _pick(plan, "cyclic")
    raw = _pick(plan, "repetitions")
    if raw is None:
        # An older plan predating `repetitions`. Derive it from `cyclic` rather than
        # falling through to the config default: a ONE-SHOT skill must be recorded
        # exactly once (recording it twice wraps phase 1 -> 0 and snaps the robot back
        # to its start pose mid-clip). A cyclic skill has no such hazard, so leave it
        # to the caller's default.
        return env, cyclic, (1 if cyclic is False else None)
    try:
        reps = int(raw)
    except (TypeError, ValueError):
        reps = 1
    return env, cyclic, max(1, min(reps, 8))
