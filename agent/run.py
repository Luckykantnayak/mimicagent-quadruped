import argparse
import json
import os

from prompts.prompt_loader import load_yaml_config, load_prompt

from langgraph.graph import StateGraph, START, END
from agents.base_agent import BaseAgent
from agents.code_gen.state import AgentState
from langchain_core.messages import AIMessage

from agents.code_gen.coder_agent import make_coder_node
from agents.code_gen.motion_planner_agent import make_motion_plan_node
from agents.code_gen.code_executer_agent import make_code_executer_node, make_finalize_node
from agents.code_gen.critic_coder_agent import make_critic_coder_node
from agents.code_gen.routers import make_critic_router, make_unit_test_router
from agents.code_gen.run_unit_test import make_unit_test_node
from agents.code_gen.skill_filter import make_skill_filter_node
from agents.code_gen.motion_diagnosis_agent import make_motion_diagnosis_agent_node
from agents.skill_gen.skill_formatter import prepare_skill

from llm.load_claude import load_llm

import time

#----------------------------------------------------
#                    LOAD CONFIG
#----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(BASE_DIR, "cfg", "config.yaml")
cfg = load_yaml_config(yaml_path)
# Per-process robot override (so a second robot can run in PARALLEL without editing
# the shared config.yaml). Everything
# downstream reads cfg["robot"]["name"], so overriding it here propagates fully.
if os.environ.get("ROBOT_NAME"):
    cfg["robot"]["name"] = os.environ["ROBOT_NAME"]

#----------------------------------------------------
#                    LOAD PROMPTS
#----------------------------------------------------

coder_prompt = load_prompt("coder", cfg)
mp_prompt = load_prompt("motion_planner", cfg)
critic_prompt = load_prompt("critic_coder", cfg)
skill_filter_prompt = load_prompt("skill_filter", cfg)
motion_diagnosis_prompt = load_prompt("motion_failure_diagnosis", cfg)

#----------------------------------------------------
#                    LOAD LLM
#----------------------------------------------------

llm = load_llm()

#----------------------------------------------------
#                    ADD NODES
#----------------------------------------------------

skill_folder = "skills_v1"

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("skill_filter_agent", lambda state: state)
    graph.add_node("motion_planner_agent", make_motion_plan_node(BaseAgent(llm), data=mp_prompt, cfg=cfg))
    graph.add_node("code_generator_agent", make_coder_node(BaseAgent(llm), data=coder_prompt, cfg=cfg))
    graph.add_node("code_executer_agent", make_code_executer_node(cfg))
    graph.add_node("unit_test_router", lambda state: state)
    graph.add_node("unit_test_agent", make_unit_test_node(cfg))
    graph.add_node("critic_router", lambda state: state)
    graph.add_node("motion_diagnosis_agent", make_motion_diagnosis_agent_node(BaseAgent(llm), data=motion_diagnosis_prompt, cfg=cfg))
    graph.add_node("critic_coder_agent", make_critic_coder_node(BaseAgent(llm), data=critic_prompt, cfg=cfg))
    graph.add_node("finalize_agent", make_finalize_node(cfg))
    graph.add_edge("finalize_agent", END)

    #----------------------------------------------------
    #                    ADD EDGES
    #----------------------------------------------------
    graph.add_edge(START, "skill_filter_agent")
    graph.add_edge("motion_planner_agent", "code_generator_agent")
    graph.add_edge("code_generator_agent", "code_executer_agent")
    graph.add_edge("code_executer_agent", "unit_test_router")
    graph.add_edge("unit_test_agent", "critic_router")
    graph.add_edge("motion_diagnosis_agent", "critic_coder_agent")
    graph.add_edge("critic_coder_agent", "code_executer_agent")

    #----------------------------------------------------
    #                ADD CONDITIONAL EDGES
    #----------------------------------------------------

    graph.add_conditional_edges(
        "skill_filter_agent",
        make_skill_filter_node(agent=BaseAgent(llm), data=skill_filter_prompt),
        {
            "continue": "motion_planner_agent",
            "end_skill": END,
        }
    )

    graph.add_conditional_edges(
        "unit_test_router",
        make_unit_test_router(cfg),
        {
            "continue": "unit_test_agent",
            "repair": "critic_coder_agent",
            "replan": "motion_planner_agent",   # repeated identical lint hard-fail = a plan bug
            "end_skill": "finalize_agent",
        }
    )

    graph.add_conditional_edges(
        "critic_router",
        make_critic_router(cfg),
        {
            "loop": "motion_diagnosis_agent",
            "replan": "motion_planner_agent",   # step back and re-decide the interface
            "end_skill": "finalize_agent",
        }
    )

    return graph.compile()


def print_stream(stream):
    terminal_state = None
    for s in stream:
        terminal_state = s
        messages = s.get("messages", [])
        if not messages:
            continue  # nothing to print

        message = messages[-1]

        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

    return terminal_state


def run_all_skills(skill_version, skills=None):
    app = build_graph()

    with open("graph_coder.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())

    all_tokens = []
    all_time = []
    all_skills = []

    for idx, skill in enumerate(skills):

        t1 = time.time()

        print("\n" + "=" * 60)
        print(f"Running skill {idx}: {skill['skill_id']}")
        print("=" * 60)

        skill_desc = {k: v for (k, v) in skill.items() if k != "think"}

        inputs = {
            "messages": [AIMessage(json.dumps(skill_desc))],
            "skill_name": skill["skill_id"],
            "skill_desc": skill_desc,
            "success": True,
            "skill_version": skill_version,
            "num_critic_iters": 0,
            "rewrite_mode": "",
            "token_usage": {},
        }

        terminal_state = print_stream(app.stream(inputs, stream_mode="values", config={"recursion_limit": 100}))

        token_usage = terminal_state["token_usage"]

        t2 = time.time()
        print("Time taken for skill:", t2 - t1)
        print("token usage:", token_usage)

        all_tokens.append(token_usage)
        all_time.append(t2 - t1)
        all_skills.append(skill)

    return all_tokens, all_time, all_skills


def save_token_usage_time(cfg, all_token_usage, all_time_taken, all_skills):
    robot = cfg["robot"]["name"]

    for token_usage, time_taken, skill in zip(all_token_usage, all_time_taken, all_skills):
        skill_dir = os.path.join(
            cfg["paths"]["skill_output_dir"], robot, skill["skill_id"]
        )
        os.makedirs(skill_dir, exist_ok=True)

        metadata = {
            "token_usage": token_usage,
            "time_taken": time_taken,
        }

        with open(os.path.join(skill_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the motion-prior synthesis pipeline.")
    parser.add_argument("--skill", help="Natural-language skill description. Runs just this skill instead of the JSON library.")
    parser.add_argument("--name", default="SKILL_CUSTOM", help="Identifier for the --skill run (used for filenames and the simulator --mode).")
    parser.add_argument("--version", default=skill_folder, help="Output/skill-library version folder.")
    args = parser.parse_args()

    if not args.skill:
        # Batch mode read a bundled input skill library (archive/), which has been
        # removed — pass a single skill with --skill "<description>" --name <NAME>.
        parser.error("--skill is required: the batch input library was removed. "
                     'Run e.g.  python run.py --skill "the robot trots forward" --name SKILL_TROT')

    formatter_prompt = load_prompt("skill_formatter", cfg)
    skill = prepare_skill(BaseAgent(llm), formatter_prompt, args.skill, name=args.name)
    print("\nSkill:\n" + json.dumps(skill, indent=2))
    skills = [skill]

    token_usage, time_taken, skills = run_all_skills(skill_version=args.version, skills=skills)
    save_token_usage_time(cfg, token_usage, time_taken, skills)
