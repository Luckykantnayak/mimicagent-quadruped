from agents.base_agent import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string
from agents.code_gen.state import AgentState
from utils.tools import extract_json_from_message
from utils.robot_spec import load_robot_spec

import os
import json
import re


def _latest_critic_payload(messages):
    """Search state messages backwards for the last unit-test critic payload (the JSON
    with motion_readout + measured failures + version_history) so the replan can SEE
    what actually happened instead of only the failing check NAMES."""
    for m in reversed(messages or []):
        try:
            obj = extract_json_from_message(get_buffer_string([m]))
        except Exception:
            continue
        if isinstance(obj, dict) and "motion_readout" in obj:
            return obj
    return None


def _prev_plan(skill_dir):
    """The most recent motion_plan_*.json (the plan being replaced) — read so the
    replan can DIFF against it rather than start blind."""
    try:
        nums = [int(re.match(r"motion_plan_(\d+)\.json", f).group(1))
                for f in os.listdir(skill_dir) if re.match(r"motion_plan_(\d+)\.json", f)]
        if nums:
            with open(os.path.join(skill_dir, f"motion_plan_{max(nums):04d}.json")) as f:
                return f.read()
    except Exception:
        pass
    return ""

def _reach_env(cfg, spec):
    """Per-limb reach envelope (robots/<name>/dynamics.json), so a replan can be shown
    exactly where the previous plan's contacts were unreachable."""
    try:
        with open(os.path.join(cfg["paths"]["robots_dir"], spec.name, "dynamics.json")) as f:
            return (json.load(f) or {}).get("reach") or {}
    except (OSError, ValueError):
        return {}


def make_motion_plan_node(motion_planner_agent: BaseAgent, data, cfg):
    spec = load_robot_spec(cfg)

    def node(state: AgentState) -> AgentState:
        system_message = data["system"]  # str
        human_message = data["human"]  # str

        context = json.dumps(state["skill_desc"])
        context_prompt = (
            "\n Here is the provided skill \n"
            "{context} \n"
            "=====Begin Implementation=====\n"
        )

        context_prompt = context_prompt.format(context=context)

        # RE-PLAN: code rewrites (or repeated identical pre-sim lint fails) stopped
        # making progress, so the last plan is suspect. `stuck` is set by run_unit_test
        # (score plateaued OR a core test stuck >=3 iters); the lint path sets a streak.
        # Capped at once. FEED it the evidence (prev plan + readout + measured failures
        # + version history) — a blind replan only tightens an already-wrong invariant.
        is_replan = not state.get("replanned", False) and (
            state.get("stuck", False) or (state.get("lint_fail_streak", 0) or 0) >= 2)
        if is_replan:
            skill_dir = os.path.join(cfg["paths"]["skill_output_dir"],
                                     cfg["robot"]["name"], state["skill_name"])
            payload = _latest_critic_payload(state.get("messages")) or {}
            prev = _prev_plan(skill_dir)
            floor_f = payload.get("physics_floor_failures") or []
            fid_f = payload.get("fidelity_failures") or []
            meas = "\n".join(
                f"  - {t.get('name')}: {t.get('feedback','')}  [metrics {json.dumps(t.get('metrics',{}))}]"
                for t in (floor_f + fid_f)) or f"  (checks: {state.get('last_failures','')})"
            hist = "\n".join(payload.get("version_history") or [])
            readout = payload.get("motion_readout", "")
            # The previous plan's OWN reach violations. Without this the planner is
            # told only that the code failed, never that ITS plan was unbuildable —
            # so it re-plans blind and reproduces the same unreachable contacts
            # (measured: a cartwheel re-plan pulled the hands inward but still left
            # 5 phases violating, because nothing ever showed it the violations).
            # This is the same evidence the CODER already receives from _plan_lint.
            reach_note = ""
            if prev:
                try:
                    from agents.code_gen.coder_agent import _plan_lint
                    reach_note = _plan_lint(prev, reach_env=_reach_env(cfg, spec)) or ""
                except Exception:
                    reach_note = ""
            context_prompt += (
                "\n=== RE-PLAN (the previous plan could not be repaired) ===\n"
                "YOUR PREVIOUS PLAN:\n" + (prev[:4000] if prev else "(unavailable)") + "\n\n"
                + (("WHERE YOUR PREVIOUS PLAN BROKE THE REACH LAW — these contacts were "
                    "unbuildable AS PLANNED, so no code could have satisfied them. Fix "
                    "these FIRST, by moving the base (in x/y AND height) or the contact:\n"
                    + reach_note + "\n\n") if reach_note.strip() else "")
                + "WHAT THE BEST ATTEMPT ACTUALLY DID (geometric readout — read it FIRST):\n"
                + (readout[:3500] if readout else "(no readout)") + "\n\n"
                "CHECKS THAT STAYED FAILING, with MEASURED values vs thresholds:\n" + meas + "\n\n"
                "VERSION HISTORY (score | floor fails | fidelity):\n" + (hist or "(none)") + "\n\n"
                "Diagnose from the MEASURED evidence above (not guesses) — one of:\n"
                " (1) WRONG CONTROL CHANNEL: a limb that must COINCIDE with a surface/object it "
                "grips/holds is on joint_posture instead of world_reach (world-IK to that point) "
                "+ an attachment. Re-decide every limb's channel.\n"
                " (2) A MIS-DECLARED FIDELITY INVARIANT: if the readout shows the motion is "
                "actually faithful but an invariant fails, the INVARIANT is wrong — re-declare it. "
                "Common: a jump/squat/inversion needs peak_rise (not net_progress-z, which is ~0 "
                "because it returns, nor a whole-trajectory base_height_band); a held object's "
                "coincidence is measured to its CENTER, so allow at least its half-extent. Compare "
                "the MEASURED value to the threshold — if the measured value is physically correct, "
                "fix the threshold/primitive (you own the invariants).\n"
                " (3) A STRUCTURAL SCHEDULE bug: a swing declared a contact BEFORE it arrives at the "
                "foothold, or a contact set that leaves a declared limb floating — retime the "
                "contact_schedule so every declared contact is actually reached and rested.\n"
                "Emit a corrected plan (channels + invariants + contact schedule + duration).\n")

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=context_prompt),
            HumanMessage(content=human_message),
        ]

        # RETRY an unparseable plan instead of dying. The plan is the FIRST node, so a
        # single malformed LLM response here kills the entire invocation having produced
        # nothing — the whole run wasted on one bad draw. Measured twice: a trailing
        # comma killed a cartwheel run 20 s in, and a response containing no JSON at all
        # ("Expecting value: line 1 column 1") killed a stairs run at 2 min. The parser
        # already tolerates the routine quirks (control characters, trailing commas);
        # this covers the rest — an empty, truncated or prose-only reply — by asking
        # again, with the requirement restated. Cheap relative to losing the run.
        prompt_text = get_buffer_string(messages)
        input_tokens = output_tokens = 0
        plan, output = None, None
        for attempt in range(3):
            ask = prompt_text if attempt == 0 else (
                prompt_text + "\n\nYour previous reply could not be parsed as JSON. "
                "Reply with the motion plan as a SINGLE valid JSON object and NOTHING "
                "else — no prose before or after, no trailing commas.")
            output = motion_planner_agent.generate(ask)   # output type: AIMessage
            usage = output.response_metadata.get('usage', {}) or {}
            input_tokens += usage.get('input_tokens', 0)
            output_tokens += usage.get('output_tokens', 0)
            try:
                plan = extract_json_from_message(get_buffer_string([output]))
                if isinstance(plan, dict) and plan:
                    break
            except (ValueError, TypeError) as e:
                print(f"[motion_planner] unparseable plan on attempt {attempt + 1}/3 ({e})")
                plan = None
        if not isinstance(plan, dict) or not plan:
            raise RuntimeError(
                "motion planner returned no parseable JSON plan after 3 attempts")

        if state["token_usage"]:
            token_dict = {
                "input_tokens": input_tokens + state["token_usage"]["input_tokens"],
                "output_tokens": output_tokens + state["token_usage"]["output_tokens"],
            }

        else:
            token_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens}

        save(skill_name=state["skill_name"],
             skill_version=state["skill_version"],
             data=plan,
             cfg=cfg
             )

        # Resolve the planner's environment choice to a full spec (scene +
        # grounding + objects). Unknown/absent -> the flat default.
        env_name = (plan or {}).get("environment") if isinstance(plan, dict) else None
        if isinstance(env_name, dict):          # planner may echo the whole block
            env_name = env_name.get("name")
        environment = spec.get_environment(env_name)

        return {
            "messages": [output],
            'skill_name': state['skill_name'],
            'skill_version': state['skill_version'],
            'success': state['success'],
            "num_critic_iters": state['num_critic_iters'],
            'rewrite_mode': state['rewrite_mode'],
            'skill_desc': state['skill_desc'],
            "token_usage": token_dict,
            "environment": environment,
            # mark the re-plan as spent (capped at once) and clear ALL stall state so
            # the fresh plan gets a clean repair budget.
            "replanned": state.get("replanned", False) or is_replan,
            "no_improve_count": 0,
            "stuck": False,
            "core_fail_streaks": {},
            "lint_fail_streak": 0,
        }

    return node


def save(skill_name: str, skill_version: str, data, cfg):
    # Motion plans live with the rest of the skill: skill_output/<robot>/<skill>/
    skill_dir = os.path.join(
        cfg["paths"]["skill_output_dir"], cfg["robot"]["name"], skill_name
    )
    os.makedirs(skill_dir, exist_ok=True)

    nums = []
    for fname in os.listdir(skill_dir):
        match = re.match(r"motion_plan_(\d+)\.json", fname)
        if match:
            nums.append(int(match.group(1)))

    next_num = max(nums) + 1 if nums else 1
    file_pth = os.path.join(skill_dir, f"motion_plan_{next_num:04d}.json")

    with open(file_pth, 'w') as f:
        json.dump(data, f, indent=4)

    return