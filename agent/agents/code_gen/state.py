from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence
from typing_extensions import TypedDict, Dict, Any
# from pydantic import BaseModel, Field

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    skill_name: str
    skill_version: str
    success: bool
    num_critic_iters: int
    rewrite_mode: str
    skill_desc: Dict[str, Any]
    token_usage: Dict[str, Any]
    # version-selection: keep the BEST code the loop ever produced, not the last
    current_code_path: str      # the .py written this iteration
    best_score: float           # score of the best version so far
    best_code_path: str         # path to the best version's .py
    last_prompt: str            # the exact prompt sent to the (critic-)coder this iteration
                                # — saved per version so the WINNING version's prompt ships
    # environment the skill runs in (resolved from the planner's choice):
    # {name, scene, grounding, objects} — drives the simulator scene + grounding
    environment: Dict[str, Any]
    # recoverability: re-decide the interface if code rewrites stop helping
    no_improve_count: int       # consecutive scored versions with no score improvement
    core_fail_streaks: Dict[str, Any]  # {core_test: consecutive-fail count} — per-test stall
    stuck: bool                 # score plateaued OR a core test stuck failing — triggers replan
    lint_fail_sig: str          # signature of the last pre-sim lint hard-fail (ee@phase)
    lint_fail_streak: int       # max per-component streak (drives the replan edge)
    lint_fail_streaks: Dict[str, Any]  # {lint component: consecutive-fail count} — a
                                # PERSISTENT component (same limb unreachable every
                                # rewrite) is a structural PLAN bug; whole-set equality
                                # misses it because incidental components flicker
    replanned: bool             # whether we've re-entered the planner (capped at once)
    replan_reason: str          # the persistently-failing checks, fed to the re-plan
    last_failures: str          # names of this iteration's failing discrete checks
    # cross-version fidelity history: one compact line per scored version (score,
    # floor failures, fidelity met) — makes oscillation visible to the critic
    readout_history: Sequence[str]