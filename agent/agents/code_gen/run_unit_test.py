import json
import os

from agents.code_gen.state import AgentState
from utils.test_motion_quality import evaluate_trajectory, save_report
from utils.trajectory_readout import summarize_trajectory
from utils.plan_io import load_plan_dict, extract_invariants
from utils.robot_spec import load_robot_spec

from langchain_core.messages import AIMessage

import pickle as pkl

# The universal PHYSICS FLOOR (always on; passing all of these is the primary term
# of the version-selection score). base_height_envelope was DEMOTED to an
# agent-declared fidelity invariant; self_collision is now SOFT (gross-only).
GENERIC_CORE = {
    "surface_penetration", "self_collision",
    "support_or_ballistic", "contact_grounded", "contact_no_slide", "joint_limits", "no_teleport",
    "ee_no_teleport", "joint_velocity_envelope", "framework_interventions",
}

# magnitude of a core violation that counts as "fully bad" (normalized -> 1.0).
# Used only to rank versions by SEVERITY when none of them pass — so a nearly-fixed
# version (8 cm penetration) outranks a badly-broken one (45 cm) that fails the
# SAME number of checks. Keys are the metric names those evaluators emit.
_SEVERITY_SCALE = {
    "surface_penetration": ("max_penetration", 0.30),
    "self_collision":      ("max_overlap", 0.15),
    "contact_no_slide":    ("max_stance_drift", 0.50),
    "no_teleport":         ("worst_step", 1.5),
    "ee_no_teleport":      ("worst_step", 1.0),
    "joint_velocity_envelope": ("worst_ratio", 5.0),
    "framework_interventions": ("clamp_max", 0.45),
}


def _core_severity(tests):
    """Mean normalized violation magnitude over the FAILED generic-core checks,
    clamped to [0, 1). A continuous 'how far from passing' signal: two versions
    that fail the same COUNT of checks are separated by HOW BADLY they fail, so
    the repair loop's progress (e.g. 45 cm -> 8 cm penetration) is not thrown away.
    Capped below 1.0 so it only ever breaks ties WITHIN a whole-number reward tier."""
    sev = []
    for t in tests:
        if t["name"] not in GENERIC_CORE or t["passed"] or t.get("skipped"):
            continue
        m = t.get("metrics", {})
        if t["name"] in _SEVERITY_SCALE:
            key, scale = _SEVERITY_SCALE[t["name"]]
            sev.append(min(1.0, float(m.get(key, scale)) / scale))
        elif t["name"] == "contact_grounded":
            # track the PASS criterion (float_frac > 0.30), not just the clearance —
            # else shaving depth looks like progress while the frac (what passes) is stuck.
            frac = (float(m.get("float_frac", 0.0)) - 0.30) / 0.70
            clr = float(m.get("float_clearance", 0.0)) / 0.30
            sev.append(min(1.0, max(0.0, frac, clr)))
        elif t["name"] == "support_or_ballistic":
            sev.append(min(1.0, 1.0 - float(m.get("rest_frac", 0.0))))
        elif t["name"] == "joint_limits":
            sev.append(min(1.0, len(m.get("violated_joints", [1])) / 5.0))
        else:
            sev.append(0.5)
    return min(0.99, sum(sev) / len(sev)) if sev else 0.0


def make_unit_test_node(cfg):

    spec = load_robot_spec(cfg)

    def run_unit_test(state: AgentState) -> AgentState:
        skill_name = state["skill_name"]
        # All artifacts for a skill live in skill_output/<robot>/<skill>/
        skill_dir = os.path.join(
            cfg["paths"]["skill_output_dir"], cfg["robot"]["name"], skill_name
        )
        fn = os.path.join(skill_dir, f"{skill_name}.pkl")

        # Load trajectory
        with open(fn, "rb") as f:
            traj = pkl.load(f)

        # Planner-owned fidelity invariants + declared sliding/pivot contacts.
        plan = load_plan_dict(skill_dir, skill_name)
        invariants, sliding_contacts, pivot_contacts = extract_invariants(plan)

        # No skill-label-derived test gates: every test's domain comes from a
        # PHYSICAL declaration in the recorded trajectory (a []-contact flight
        # window, scene objects present) or the planner's explicit invariants.
        dt = 1.0 / traj["fps"]
        feedback = evaluate_trajectory(
            traj=traj, spec=spec, dt=dt,
            invariants=invariants, sliding_contacts=sliding_contacts,
            pivot_contacts=pivot_contacts,
            cyclic=(plan or {}).get("cyclic"),
        )
        summary = feedback["summary"]

        # PERCEPTION BEFORE VERDICT: a geometric readout of what the motion ACTUALLY
        # did (base path, per-limb contact timeline + lift + drift, gait phase
        # offsets, object path). Fed to diagnosis + critic so the loop reasons about
        # the MOTION, not just the pass/fail lamps — the fix for the blindness that
        # made the checks oscillate.
        readout = summarize_trajectory(traj, spec, dt)

        # The loop continues while EITHER the physics floor OR the fidelity layer
        # is unmet — fidelity must DRIVE repair, not merely rank versions. Without
        # the fidelity tier here, the loop exited the instant the floor was clean
        # (a carrybox that had picked the box up but let it drift from the hands
        # shipped at 4/6 because passed_all_discrete was already true).
        fid_unmet = summary.get("num_fidelity_passed", 0) < summary.get("num_fidelity_total", 0)
        if not summary["passed_all_discrete"]:
            rewrite_mode = "structural"
        elif fid_unmet:
            rewrite_mode = "fidelity"     # floor clean, but it doesn't yet LOOK like the skill
        else:
            rewrite_mode = "none"

        # Extract failures
        discrete_failures = [
            t for t in feedback["tests"]
            if t["category"] == "discrete"
               and not t["passed"]
               and not t["skipped"]
        ]
        fidelity_failures = [
            t for t in feedback["tests"]
            if t["category"] == "fidelity"
               and not t["passed"]
               and not t["skipped"]
        ]
        # Form feedback for the diagnosis + critic agents (they read state["messages"]).
        critic_payload = dict(
            rewrite_mode=rewrite_mode,
            motion_readout=readout,        # <-- what the robot ACTUALLY did

            summary=dict(
                passed_all=summary["passed_all"],
                passed_all_discrete=summary["passed_all_discrete"],
                total_reward=summary["total_reward"],
                fidelity=f"{summary.get('num_fidelity_passed',0)}/{summary.get('num_fidelity_total',0)} invariants met",
            ),

            physics_floor_failures=[
                {"name": t["name"], "metrics": t["metrics"], "feedback": t["feedback"]}
                for t in discrete_failures
            ],
            fidelity_failures=[
                {"name": t["name"], "metrics": t["metrics"], "feedback": t["feedback"]}
                for t in fidelity_failures
            ],
        )

        os.makedirs(skill_dir, exist_ok=True)
        save_report(feedback, skill_name="report", out_dir=skill_dir)

        # --- VERSION SELECTION: score this version; keep the BEST code so far ---
        # Tiers (each strictly dominates the next): physics floor pass (1000) >>
        # all-discrete pass incl. object checks (100) >> fidelity fraction met
        # (<=40, ranks floor-passing versions by how well they match the intended
        # motion) >> reward (integer, fewer/lesser failures) >> -core_severity
        # (<1, how badly the floor fails). An invariant can NEVER outrank a floor
        # fail (40 < 100 < 1000).
        generic_pass = all(
            t["passed"] for t in feedback["tests"] if t["name"] in GENERIC_CORE
        )
        fid_total = max(1, summary.get("num_fidelity_total", 0))
        fid_frac = summary.get("num_fidelity_passed", 0) / fid_total
        # The fidelity weight is GATED ON THE FLOOR. The tier comment says an invariant
        # can never outrank a floor failure "(40 < 100 < 1000)" — but that only holds
        # once some version PASSES the floor and banks the +100. Among versions that all
        # FAIL it, a flat 40*fid_frac swamps `reward`, which moves only ~1 point per
        # failed check, so a physically WORSE motion can win on looks. Measured live on a
        # ledge-drop: v5 (3 floor failures, fidelity 5/7) scored 24.6 and shipped over
        # v6 (ONE floor failure, 4/7) at 20.4. While the floor is broken, fidelity may
        # only BREAK TIES; it earns its full weight once the physics is actually right.
        #
        # The gate is the PHYSICS FLOOR (`generic_pass`), NOT `passed_all_discrete`.
        # `passed_all_discrete` also contains the TASK checks (object_grip,
        # object_continuity, flight_window) — which measure the same thing the fidelity
        # tier does: did the skill actually DO the thing. Keying on it meant that
        # FAILING THE TASK SUPPRESSED THE VERY TERM THAT MEASURES THE TASK, a feedback
        # loop with the wrong sign. Measured on the carry-box: physics floor clean
        # (+1000), `object_grip` failed, so fidelity was weighted 2.0 and contributed
        # 1.1 points out of ~1000 — the loop had no incentive to ever pick the box up,
        # and rationally shipped a physically-valid motion that never grips it.
        fid_weight = 40.0 if generic_pass else 2.0
        score = (summary["total_reward"]
                 + fid_weight * fid_frac
                 + (100.0 if summary["passed_all_discrete"] else 0.0)
                 + (1000.0 if generic_pass else 0.0)
                 - _core_severity(feedback["tests"]))
        prev_best = state.get("best_score")
        cur_path = state.get("current_code_path")
        improved = prev_best is None or score > prev_best
        if improved or score >= prev_best:
            best_score, best_code_path = score, cur_path
        else:
            best_score, best_code_path = prev_best, state.get("best_code_path")
        # No early stop-loss: with fidelity DRIVING the loop, a version that clears
        # the floor but scores below a prior best (e.g. a fidelity fix that briefly
        # regressed something) must still be allowed to keep experimenting toward
        # higher fidelity. Termination is handled by the hard iteration cap, the
        # one-shot replan when stuck, and finalize (which always re-instates the
        # best-scoring version) — so a late regression can never ship.

        # Stuck detector (for the re-plan path): count consecutive scored versions
        # that did NOT improve the score.
        no_improve = 0 if improved else (state.get("no_improve_count") or 0) + 1
        last_failures = ", ".join(
            t["name"] for t in discrete_failures + fidelity_failures
        )
        # PER-TEST stall: the OVERALL score can tick up (a severity change on an
        # unrelated check) while a SPECIFIC core test stays stuck failing — so
        # no_improve resets and the loop never replans, even though one check has been
        # dead for iterations (carrybox's object_grip, crawl's contact_grounded ×6). A
        # whole-SET signature never fires either (fidelity churn resets it every
        # version). Track per-CORE-test consecutive-fail STREAKS: a stall = any core
        # test failing >= 3 versions in a row.
        core_now = {t["name"] for t in discrete_failures if t["name"] in GENERIC_CORE}
        prev_streaks = state.get("core_fail_streaks") or {}
        core_fail_streaks = {n: prev_streaks.get(n, 0) + 1 for n in core_now}
        per_test_stall = any(v >= 3 for v in core_fail_streaks.values())
        # STUCK — the single flag BOTH the critic-router (replan decision) and the
        # planner (replan context) read, so they can never disagree: the overall score
        # plateaued for >=2 iters, OR a core test has been stuck failing for >=3.
        stuck = (no_improve >= 2) or per_test_stall

        # Cross-version readout history (a compact oscillation-visible table). The
        # 5-iteration crawl thrash was invisible frame-by-frame and obvious across
        # versions — give the critic the whole trajectory of attempts.
        it = (state.get("num_critic_iters") or 0)
        floor_fail = ",".join(t["name"] for t in discrete_failures) or "floor OK"
        fid_note = f"{summary.get('num_fidelity_passed',0)}/{summary.get('num_fidelity_total',0)} fidelity"
        hist_line = (f"v{it}: score {score:.1f} | {floor_fail} | {fid_note} | "
                     f"reward {summary['total_reward']:.1f}")
        readout_history = list(state.get("readout_history") or []) + [hist_line]
        critic_payload["version_history"] = readout_history

        return {
            "messages": [AIMessage(json.dumps(critic_payload))],
            'skill_name': state['skill_name'],
            'skill_version': state['skill_version'],
            'success': state['success'],
            "num_critic_iters": (1 if state['num_critic_iters'] is None else state['num_critic_iters'] + 1),
            'rewrite_mode': rewrite_mode,
            'skill_desc': state['skill_desc'],
            'best_score': best_score,
            'best_code_path': best_code_path,
            'no_improve_count': no_improve,
            'core_fail_streaks': core_fail_streaks,
            'stuck': stuck,
            'last_failures': last_failures,
            'readout_history': readout_history,
        }

    return run_unit_test
