import os
import json

import streamlit as st
from langchain_core.messages import AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END

from prompts.prompt_loader import load_yaml_config, load_prompt
from llm.load_claude import load_llm
from agents.base_agent import BaseAgent
from agents.code_gen.state import AgentState
from agents.code_gen.coder_agent import make_coder_node
from agents.code_gen.motion_planner_agent import make_motion_plan_node
from agents.code_gen.code_executer_agent import make_code_executer_node
from agents.code_gen.critic_coder_agent import make_critic_coder_node
from agents.code_gen.routers import make_critic_router, should_continue
from agents.code_gen.run_unit_test import make_unit_test_node
from agents.code_gen.skill_filter import make_skill_filter_node
from agents.code_gen.motion_diagnosis_agent import make_motion_diagnosis_agent_node
from agents.skill_gen.skill_formatter import prepare_skill, is_structured_skill

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE_DIR, "cfg", "config.yaml")

# Friendly labels for the nodes that actually produce output.
NODE_LABELS = {
    "motion_planner_agent": "Motion Planner",
    "code_generator_agent": "Coder",
    "code_executer_agent": "Code Executor + Simulator",
    "unit_test_agent": "Unit Test / Trajectory Quality",
    "motion_diagnosis_agent": "Motion Diagnosis",
    "critic_coder_agent": "Critic (rewrite)",
}


def build_app(llm, cfg):
    coder_prompt = load_prompt("coder", cfg)
    mp_prompt = load_prompt("motion_planner", cfg)
    critic_prompt = load_prompt("critic_coder", cfg)
    skill_filter_prompt = load_prompt("skill_filter", cfg)
    motion_diagnosis_prompt = load_prompt("motion_failure_diagnosis", cfg)

    graph = StateGraph(AgentState)
    graph.add_node("skill_filter_agent", lambda state: state)
    graph.add_node("motion_planner_agent", make_motion_plan_node(BaseAgent(llm), data=mp_prompt, cfg=cfg))
    graph.add_node("code_generator_agent", make_coder_node(BaseAgent(llm), data=coder_prompt))
    graph.add_node("code_executer_agent", make_code_executer_node(cfg))
    graph.add_node("unit_test_router", lambda state: state)
    graph.add_node("unit_test_agent", make_unit_test_node(cfg))
    graph.add_node("critic_router", lambda state: state)
    graph.add_node("motion_diagnosis_agent", make_motion_diagnosis_agent_node(BaseAgent(llm), data=motion_diagnosis_prompt, cfg=cfg))
    graph.add_node("critic_coder_agent", make_critic_coder_node(BaseAgent(llm), data=critic_prompt, cfg=cfg))

    graph.add_edge(START, "skill_filter_agent")
    graph.add_edge("motion_planner_agent", "code_generator_agent")
    graph.add_edge("code_generator_agent", "code_executer_agent")
    graph.add_edge("code_executer_agent", "unit_test_router")
    graph.add_edge("unit_test_agent", "critic_router")
    graph.add_edge("motion_diagnosis_agent", "critic_coder_agent")
    graph.add_edge("critic_coder_agent", "code_executer_agent")

    graph.add_conditional_edges(
        "skill_filter_agent",
        make_skill_filter_node(agent=BaseAgent(llm), data=skill_filter_prompt),
        {"continue": "motion_planner_agent", "end_skill": END},
    )
    graph.add_conditional_edges(
        "unit_test_router",
        should_continue,
        {"continue": "unit_test_agent", "end_skill": END},
    )
    graph.add_conditional_edges(
        "critic_router",
        make_critic_router(cfg),
        {"loop": "motion_diagnosis_agent", "end_skill": END},
    )

    return graph.compile()


def last_message_text(update):
    messages = update.get("messages", []) if isinstance(update, dict) else []
    if not messages:
        return ""
    return get_buffer_string([messages[-1]])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MimicAgent", layout="wide")
st.title("MimicAgent — Text → Motion Prior")
st.caption("Describe a skill in natural language and watch the agents plan, code, execute, and repair it.")

cfg = load_yaml_config(CFG_PATH)
robots_dir = cfg["paths"]["robots_dir"]
robots = sorted(
    d for d in os.listdir(robots_dir)
    if os.path.isfile(os.path.join(robots_dir, d, "robot.yaml"))
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Anthropic API key", value=os.environ.get("ANTHROPIC_API_KEY", ""), type="password")
    robot = st.selectbox("Robot", robots, index=robots.index(cfg["robot"]["name"]) if cfg["robot"]["name"] in robots else 0)
    skill_name = st.text_input("Skill name (identifier)", "SKILL_CUSTOM")
    version = st.text_input("Version (output folder)", "demo")
    max_iters = st.number_input("Max critic iterations", min_value=0, max_value=10, value=int(cfg["critic"]["max_iters"]))

description = st.text_area(
    "Skill description",
    "The robot trots forward at a steady pace using a symmetric diagonal gait.",
    height=140,
)

run = st.button("Run", type="primary")

if run:
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    os.environ["ANTHROPIC_API_KEY"] = api_key
    cfg["robot"]["name"] = robot
    cfg["critic"]["max_iters"] = int(max_iters)

    try:
        llm = load_llm()
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.stop()

    # Step 0: prepare the skill. Structured JSON is used as-is; raw text is
    # converted by the formatter agent.
    already_structured = is_structured_skill(description)
    step_label = "Using provided skill" if already_structured else "Formatting skill from your description…"
    with st.status(step_label, expanded=True) as fmt_status:
        try:
            formatter_prompt = load_prompt("skill_formatter", cfg)
            skill_desc = prepare_skill(BaseAgent(llm), formatter_prompt, description, name=skill_name)
            st.json(skill_desc)
            fmt_status.update(
                label="Skill ready" if already_structured else "Skill formatted",
                state="complete",
            )
        except Exception as e:
            fmt_status.update(label="Skill preparation failed", state="error")
            st.exception(e)
            st.stop()

    inputs = {
        "messages": [AIMessage(json.dumps(skill_desc))],
        "skill_name": skill_desc["skill_id"],
        "skill_desc": skill_desc,
        "success": True,
        "skill_version": version,
        "num_critic_iters": 0,
        "rewrite_mode": "",
        "token_usage": {},
    }

    app = build_app(llm, cfg)

    st.subheader("Live execution")
    terminal_state = {}
    step_idx = 0

    with st.status("Running pipeline…", expanded=True) as status:
        try:
            for step in app.stream(inputs, stream_mode="updates", config={"recursion_limit": 100}):
                for node, update in step.items():
                    if isinstance(update, dict) and "success" in update:
                        terminal_state = update  # nodes return the full state; keep the latest

                    step_idx += 1
                    label = NODE_LABELS.get(node)
                    if label is None:
                        # Pass-through / routing nodes: show a thin breadcrumb only.
                        st.caption(f"→ routing: {node}")
                        continue

                    text = last_message_text(update)
                    with st.expander(f"{step_idx}. {label}", expanded=True):
                        if node == "code_generator_agent":
                            st.code(text, language="python")
                        elif node == "unit_test_agent":
                            try:
                                st.json(json.loads(text.split("AI:", 1)[-1].strip()))
                            except Exception:
                                st.write(text)
                        else:
                            st.write(text)
            status.update(label="Pipeline finished", state="complete")
        except Exception as e:
            status.update(label="Pipeline error", state="error")
            st.exception(e)

    st.subheader("Result")
    cols = st.columns(3)
    cols[0].metric("Success", str(terminal_state.get("success", "—")))
    cols[1].metric("Rewrite mode", terminal_state.get("rewrite_mode") or "—")
    cols[2].metric("Critic iterations", terminal_state.get("num_critic_iters", "—"))
    if terminal_state.get("token_usage"):
        st.caption("Token usage")
        st.json(terminal_state["token_usage"])
    st.caption(
        "Generated code is written to `skills/<robot>/<skill name>/`; the motion plan and "
        "quality report are written under `data/` (per the paths in `cfg/config.yaml`)."
    )
