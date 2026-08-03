import os
import re
import json

import subprocess

from langchain_core.messages import get_buffer_string, AIMessage
from agents.code_gen.state import AgentState
from utils.tools import extract_python_code_from_message

from typing import Dict, Any


def make_code_executer_node(cfg):
    def CodeExecuterAgent(state: AgentState)->AgentState:
        """
        This agent is responsible for:
            1. Generating {skill}.py file in the Data Generation codebase
            2. Running the simulator
            3. Determining success: Store Yes if no bugs, else Store No and the execution error in the state
        :param state:
        :return:
        """
        try:
            # coder_output = json.loads(output)
            output = extract_python_code_from_message(get_buffer_string([state['messages'][-1]]))
            skill_class = state["skill_name"]

        except Exception as e:
            ai_message = AIMessage([f"Error parsing JSON output.\n Error: {e} \n Fix the error and try again."])
            return {
                'messages': [ai_message],
                'skill_name': state['skill_name'],
                'skill_version': state['skill_version'],
                'success': False,
                'num_critic_iters': state['num_critic_iters'] + 1,
                'rewrite_mode': state['rewrite_mode'],
                'skill_desc': state['skill_desc'],

            }

        message = write_to_file(state, output, cfg)

        if not message['success']:
            return {
                'messages': [message['messages']],
                'skill_name': state['skill_name'],
                'skill_version': state['skill_version'],
                'success': False,
                'num_critic_iters': state['num_critic_iters'] + 1,
                'rewrite_mode': state['rewrite_mode'],
                'skill_desc': state['skill_desc'],

            }

        # PRE-SIM REACHABILITY LINT (grounding "off" only): a fast, no-IK geometry
        # check that the declared world contacts are within limb reach from the
        # authored base. A contact beyond reach would FLOAT once the sim runs — so
        # catch it here (milliseconds) with a precise base-coordinate fix and route
        # straight back to the coder, INSTEAD of spending a minutes-long rollout on
        # a doomed skill. Only a HARD violation (one the runtime base-reach seat
        # cannot cleanly absorb) short-circuits; smaller ones fall through to the
        # sim where the seat handles them.
        # A skill planned for a scene this robot does not have would otherwise run on a
        # bare floor and can score a clean PASS while doing nothing it was asked to do
        # (its object code animates a body that does not exist, and every object check
        # self-skips because the scene has no objects). Fail it into the repair loop
        # instead, naming the asset to add.
        _env = state.get('environment') or {}
        if _env.get("missing"):
            return {
                'messages': [AIMessage(
                    f"MISSING ASSET — no simulation was run. The plan asked for "
                    f"environment '{_env.get('requested')}', which is not registered for "
                    f"this robot, so the skill would run on a bare floor with none of the "
                    f"objects it expects. Re-plan using ONE of the registered "
                    f"environments, or ask the user to add "
                    f"assets/scenes/{_env.get('requested')}.xml and register it under "
                    f"`environments:` in the robot spec.")],
                'current_code_path': message['file_path'],
                'skill_name': state['skill_name'],
                'skill_version': state['skill_version'],
                'success': False,
                'num_critic_iters': state['num_critic_iters'] + 1,
                'rewrite_mode': state['rewrite_mode'],
                'skill_desc': state['skill_desc'],
                # A structural PLAN fault, not a code fault: streak it so a repeat routes
                # to the planner (replan) rather than looping the coder.
                'lint_fail_sig': f"missing_env:{_env.get('requested')}",
                'lint_fail_streak': (state.get("lint_fail_streak", 0) or 0) + 1,
            }

        lint = run_reach_lint(skill_class, cfg, environment=state.get('environment'))
        if lint and lint.get("hard_fail"):
            # Signature of WHAT the lint caught, so REPEATED IDENTICAL lint fails read
            # as a structural PLAN bug (the unit_test_router then REPLANS instead of
            # looping the coder on a schedule the code can't fix).
            # SIGNATURE = WHICH limbs failed WHICH WAY — deliberately COARSE.
            # Its only job is to answer "is the coder failing the same way again?", so
            # that a repeat escalates to a REPLAN instead of looping on a schedule the
            # code cannot fix. It must therefore NOT include the phase: the phase is a
            # float that shifts a little on every rewrite, so an exact-match streak
            # could never accumulate and the replan edge was unreachable in practice
            # (observed live: 4 consecutive lint hard-fails, 0 replans, 0 simulations).
            sig_parts = [f"arr:{ee}" for ee in sorted(lint.get("arrivals") or {})]
            if lint.get("incompatible"):
                sig_parts.append("incompat")
            sig_parts += [f"horiz:{ee}" for ee, v in sorted((lint.get("contacts") or {}).items())
                          if v.get("horizontal")]
            sig_parts += [f"notgt:{ee}" for ee in sorted(lint.get("notarget") or {})]
            # A chronically out-of-reach FREE limb is a hard-fail reason too, so it must
            # contribute to the signature — otherwise a run failing ONLY for that reason
            # produces an EMPTY signature, which never matches and never streaks.
            sig_parts += [f"free:{ee}" for ee, v in sorted((lint.get("free_reach") or {}).items())
                          if v.get("severe_frac", 0) > 0]
            sig = ";".join(sig_parts)
            # PER-COMPONENT streaks, not whole-set equality. The persistent fault is
            # usually a stable CORE (the same limb out of reach every time) wrapped in
            # INCIDENTAL components that flicker as the coder rewrites — an arrival
            # teleport here, a missing target there. Measured over one real run: the
            # core `horiz:{left,right}_hand;free:{left,right}_hand` was present in ALL 8
            # versions, yet exact whole-set equality first matched only at v7/v8, by
            # which point the iteration budget was spent and `end_skill` won the router
            # race — so the replan never fired despite the same fault 8 times running.
            # This mirrors `core_fail_streaks` in run_unit_test.py, which hit and solved
            # the identical problem on the quality-check side.
            prev = state.get("lint_fail_streaks") or {}
            streaks = {c: prev.get(c, 0) + 1 for c in sig_parts}
            streak = max(streaks.values()) if streaks else 0
            return {
                'messages': [AIMessage(
                    "PRE-SIM LINT FAILED — no simulation was run; fix this first.\n"
                    + lint.get("report", ""))],
                'current_code_path': message['file_path'],
                'skill_name': state['skill_name'],
                'skill_version': state['skill_version'],
                'success': False,
                'num_critic_iters': state['num_critic_iters'] + 1,
                'rewrite_mode': state['rewrite_mode'],
                'skill_desc': state['skill_desc'],
                'lint_fail_sig': sig,
                'lint_fail_streaks': streaks,
                'lint_fail_streak': streak,
            }

        result = run_skill(skill_class, skill_folder_version=state['skill_version'],
                           cfg=cfg, environment=state.get('environment'))

        if result['success']:
            return {
                'messages': [AIMessage("Success")],
                'current_code_path': message['file_path'],
                'skill_name': state['skill_name'],
                'skill_version': state['skill_version'],
                'success': True,
                'num_critic_iters': state['num_critic_iters'],
                'rewrite_mode': state['rewrite_mode'],
                'skill_desc': state['skill_desc'],
                'lint_fail_streak': 0,   # a clean sim run breaks the lint-fail streak
                'lint_fail_streaks': {},

            }

        else:
            return {
                'messages': [AIMessage('Failure\n' + result['stderr'] + '\n' + result['exception'])],
                'current_code_path': message['file_path'],
                'skill_name': state['skill_name'],
                'skill_version': state['skill_version'],
                'success': False,
                'num_critic_iters': state['num_critic_iters'] + 1,
                'rewrite_mode': state['rewrite_mode'],
                'skill_desc': state['skill_desc'],

            }

    return CodeExecuterAgent


def make_finalize_node(cfg):
    """Version-selection: at loop end, re-instate the BEST code the loop produced
    (not the last) and DETERMINISTICALLY re-run the sim so the shipped pkl matches
    the shipped code. The skill loader always loads the highest-numbered file, so
    the best is copied to a new highest number. We ALWAYS re-run the shipped
    version under the resolved environment and VERIFY it succeeded — a silent
    finalize failure is how a stale pkl (a different version's trajectory) ends up
    shipping alongside the wrong code."""
    import re
    import shutil

    def finalize(state: AgentState) -> AgentState:
        best = state.get("best_code_path")
        skill_name = state["skill_name"]
        environment = state.get("environment")
        skill_dir = os.path.join(
            cfg["paths"]["skill_output_dir"], cfg["robot"]["name"], skill_name)
        if not best or not os.path.exists(best):
            print(f"[finalize] no best_code_path recorded for {skill_name}; leaving as-is")
            return {}
        pat = re.compile(rf"{re.escape(skill_name)}_(\d+)\.py$")
        vers = [(int(m.group(1)), f) for f in os.listdir(skill_dir)
                if (m := pat.match(f))]
        if not vers:
            return {}
        maxn, maxf = max(vers)

        # Make the BEST version the highest-numbered one (what the loader ships).
        if os.path.join(skill_dir, maxf) != best:
            shipped = os.path.join(skill_dir, f"{skill_name}_{maxn + 1:04d}.py")
            shutil.copyfile(best, shipped)
            print(f"[finalize] re-instating best {os.path.basename(best)} "
                  f"-> {os.path.basename(shipped)}")
        else:
            shipped = os.path.join(skill_dir, maxf)
            print(f"[finalize] best is already the latest ({os.path.basename(best)})")

        # ALWAYS re-run the shipped version under the resolved environment so the
        # recorded pkl is deterministically this exact code's trajectory.
        result = run_skill(skill_name, skill_folder_version=state.get("skill_version"),
                           cfg=cfg, environment=environment)
        if not result.get("success"):
            print(f"[finalize] WARNING: re-run of the shipped skill FAILED — the "
                  f"recorded {skill_name}.pkl may NOT match {os.path.basename(shipped)}.\n"
                  f"{result.get('stderr', '')}\n{result.get('exception', '')}")
        else:
            print(f"[finalize] shipped {os.path.basename(shipped)}; "
                  f"pkl re-recorded under environment "
                  f"'{(environment or {}).get('name', 'flat')}'")
            # Render a forward-playing preview GIF of the shipped skill into its dir
            # (never blocks/fails the run — a GIF is a preview, not an artifact the
            # pipeline depends on).
            render_skill_gif(skill_name, environment, cfg)

        # --- SHIP ONLY THE FINAL VERSION (+ its prompt + its plan), clean the rest ---
        # The user wants the output dir to keep ONLY the winning skill .py, the exact
        # PROMPT given to the framework for it, and the motion PLANNER's plan — not the
        # per-iteration intermediates. Runs AFTER the re-run + GIF (which need the
        # versioned file); the loader now prefers the clean <skill>.py.
        try:
            import glob
            # final code
            shutil.copyfile(shipped, os.path.join(skill_dir, f"{skill_name}.py"))
            # final prompt: look it up by the ORIGINAL best version's number (`best`), NOT
            # the re-instated `shipped` number — re-instating copies only the .py to a new
            # number, so that new number has no prompt file; the best's own number does.
            wp = None
            mb = re.search(rf"{re.escape(skill_name)}_(\d+)\.py$", best)
            if mb:
                cand = os.path.join(skill_dir, f"{skill_name}_{int(mb.group(1)):04d}_prompt.md")
                if os.path.isfile(cand):
                    wp = cand
            if wp is None:                       # fallback: the latest prompt file present
                pf = sorted(glob.glob(os.path.join(skill_dir, f"{skill_name}_[0-9]*_prompt.md")))
                wp = pf[-1] if pf else None
            if wp and os.path.isfile(wp):
                shutil.copyfile(wp, os.path.join(skill_dir, f"{skill_name}_prompt.md"))
            # final motion plan (latest the planner produced)
            plans = sorted(glob.glob(os.path.join(skill_dir, "motion_plan_*.json")))
            if plans:
                shutil.copyfile(plans[-1], os.path.join(skill_dir, f"{skill_name}_plan.json"))
            # remove the per-iteration intermediates (keep final code/prompt/plan + artifacts)
            for pat in (f"{skill_name}_[0-9]*.py", f"{skill_name}_[0-9]*_prompt.md",
                        "motion_plan_*.json", "report_*.json"):
                for f in glob.glob(os.path.join(skill_dir, pat)):
                    os.remove(f)
            print(f"[finalize] shipped final {skill_name}.py + {skill_name}_prompt.md + "
                  f"{skill_name}_plan.json; cleaned intermediates.")
        except Exception as e:
            print(f"[finalize] cleanup/ship-final skipped ({e})")
        return {}

    return finalize


def render_skill_gif(skill_name, environment, cfg):
    """Render a forward-looping preview GIF of the shipped skill into its output dir
    (skill_output/<robot>/<skill>/<skill>.gif) via the offscreen record_video tool.
    Best-effort: any failure is logged and swallowed (the GIF is a convenience)."""
    env = environment or {}
    scene = env.get("scene")
    grounding = env.get("grounding", "contact")
    if isinstance(grounding, bool):
        grounding = "off" if grounding is False else "contact"
    robot = cfg["robot"]["name"]
    out = os.path.join(cfg["paths"]["skill_output_dir"], robot, skill_name, f"{skill_name}.gif")
    cmd = [cfg["simulator"]["python_bin"], "simulator/tools/record_video.py",
           "--robot", robot, "--mode", skill_name,
           "--grounding", str(grounding), "--cycles", "1", "--fps", "33",
           "--width", "640", "--height", "360",   # a light preview GIF, not full-res
           "--out", out]
    if scene:
        cmd += ["--scene", scene]
    try:
        r = subprocess.run(cmd, cwd=cfg["paths"]["simulator_dir"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            print(f"[finalize] preview GIF -> {out}")
        else:
            print(f"[finalize] GIF render skipped (rc={r.returncode}): "
                  f"{(r.stderr or '').strip().splitlines()[-1] if r.stderr else ''}")
    except Exception as e:
        print(f"[finalize] GIF render skipped ({e})")


def write_to_file(state: AgentState, code: str, cfg: Dict[str, Any]):
    """
    Write generated skill code using versioned filenames:
    <skill_name>_0001.py, <skill_name>_0002.py, ...

    Returns success + error message if any.
    """

    ai_message = []

    skill_name = state["skill_name"]
    # Generated skills are saved per robot: skill_output/<robot>/<skill_name>/
    skill_dir = os.path.join(
        cfg["paths"]["skill_output_dir"], cfg["robot"]["name"], skill_name
    )

    os.makedirs(skill_dir, exist_ok=True)

    # --------------------------------------------------
    # Find existing versions
    # --------------------------------------------------
    pattern = re.compile(rf"{skill_name}_(\d+)\.py$")
    versions = []

    for fname in os.listdir(skill_dir):
        m = pattern.match(fname)
        if m:
            versions.append(int(m.group(1)))

    next_version = max(versions) + 1 if versions else 1
    fname = f"{skill_name}_{next_version:04d}.py"
    file_path = os.path.join(skill_dir, fname)

    # --------------------------------------------------
    # Write file
    # --------------------------------------------------
    try:
        with open(file_path, "w") as f:
            f.write(code)

        # Save the EXACT prompt that produced this version alongside it, so the
        # WINNING version's prompt can be shipped at finalize (the user asked to keep
        # the prompt given to the framework). Best-effort — never fails the write.
        prompt = state.get("last_prompt")
        if prompt:
            try:
                with open(os.path.join(skill_dir, f"{skill_name}_{next_version:04d}_prompt.md"), "w") as pf:
                    pf.write(f"# Coder prompt for {skill_name} v{next_version:04d}\n\n" + prompt)
            except OSError:
                pass

        success = True

    except Exception as e:
        ai_message = [
            AIMessage(
                content=(
                    "Failed to write python code to file.\n"
                    f"Error: {e}\n"
                    "Fix the error and try again."
                )
            )
        ]
        success = False

    return {
        "messages": ai_message,
        "success": success,
        "file_path": file_path if success else None,
        "version": next_version if success else None,
    }


def run_reach_lint(skill_name: str, cfg, environment=None):
    """Pre-sim reachability lint (fast, no IK/rendering). Returns the parsed report
    dict, or None if it does not apply (grounding != off) or could not run — in
    which case the caller just proceeds to the sim. Never raises."""
    env = environment or {}
    grounding = env.get("grounding", "contact")
    if isinstance(grounding, bool):
        grounding = "off" if grounding is False else "contact"
    # The lint now runs in BOTH grounding modes: the ARRIVAL check (a swing that
    # teleports into contact) applies to any skill; the REACHABILITY check inside
    # is self-gated to grounding "off". So we no longer skip on grounding.
    cmd = [
        cfg["simulator"]["python_bin"],
        "simulator/tools/reach_lint.py",
        "--mode", skill_name,
        "--robot", cfg["robot"]["name"],
        "--grounding", str(grounding),
    ]
    if env.get("scene"):
        cmd += ["--scene", env["scene"]]
    try:
        result = subprocess.run(cmd, cwd=cfg["paths"]["simulator_dir"],
                                capture_output=True, text=True, timeout=180)
        for line in (result.stdout or "").splitlines():
            if line.startswith("LINT_REPORT_JSON:"):
                return json.loads(line.split("LINT_REPORT_JSON: ", 1)[1])
    except Exception as e:
        print(f"[reach-lint] skipped ({e}); proceeding to sim")
    # A lint that produced NO report did not "pass" — it FAILED TO RUN, and the two
    # are opposite in meaning while looking identical downstream (both return None and
    # the rollout proceeds). Fail-open is the right robustness choice: a bug in a
    # pre-flight check must never block generation. But it must be LOUD, because a
    # silently-disabled safety check is worse than no safety check — the loop, the
    # logs and the reader all believe the skill was screened when it was not.
    # Measured: a TypeError inside the surface-existence check disabled the ENTIRE
    # lint, and a ledge-drop that stood 0.75 m up on empty air sailed through to the
    # sim and shipped, while the run log said only "proceeding to sim".
    if result_seen := locals().get("result"):
        err = (result_seen.stderr or "").strip().splitlines()
        tail = err[-1] if err else "no LINT_REPORT_JSON on stdout"
        print(f"⚠️  [reach-lint] PRODUCED NO REPORT — the pre-sim screen did NOT run for "
              f"this version; treat it as UNSCREENED, not as passing. Last stderr: {tail}")
    return None


def run_skill(skill_name: str, skill_folder_version: str, cfg, environment=None):
    python_bin = cfg["simulator"]["python_bin"]

    # Environment: scene + grounding chosen by the planner (default = flat floor,
    # contact grounding). A scene with objects animates them via the skill's
    # compute_object_poses; grounding "off" lets a skill set an absolute base path.
    env = environment or {}
    scene = env.get("scene")
    grounding = env.get("grounding", "contact")
    # YAML parses an unquoted `off`/`on` as a bool — normalize back to a mode name.
    if isinstance(grounding, bool):
        grounding = "off" if grounding is False else "contact"

    # REPETITIONS: a cyclic skill authors ONE gait cycle (self.freq = 1/period) and the
    # framework records `repetitions` cycles — so a walk travels several strides without
    # stretching one cycle over the whole duration (which drags each stance past its
    # foothold; the WALK4/BOUND float). Read the planner's declared repetitions from the
    # latest motion plan; default 1 (one-shot skills), clamp to a sane ceiling.
    record_cycles = 1
    try:
        import glob
        skill_dir = os.path.join(cfg["paths"]["skill_output_dir"], cfg["robot"]["name"], skill_name)
        plans = sorted(glob.glob(os.path.join(skill_dir, "motion_plan_*.json")))
        if plans:
            _pj = json.load(open(plans[-1]))
            _mp = _pj.get("motion_plan", _pj)
            record_cycles = int(_mp.get("repetitions") or _pj.get("repetitions") or 1)
            record_cycles = max(1, min(record_cycles, 8))
    except Exception:
        pass

    cmd = [
        python_bin,
        "simulator/run.py",
        "--mode", skill_name,
        "--robot", cfg["robot"]["name"],
        "--record", "True",
        "--headless", "True",
        "--grounding", str(grounding),
        "--record_cycles", str(record_cycles),   # planner-declared cycle count (default 1)
    ]
    if scene:
        cmd += ["--scene", scene]

    try:
        result = subprocess.run(
            cmd,
            cwd=cfg["paths"]["simulator_dir"],
            check=True,
            capture_output=True,
            text=True,
        )

        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "returncode": e.returncode,
            "exception": str(e),
        }

