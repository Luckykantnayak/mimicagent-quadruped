from prompts.build_json_prompts import load_yaml_config, load_prompt

from langgraph.graph import StateGraph, START, END
from agents.skill_gen.state import AgentState
from agents.skill_gen.skill_router import build_skill_router_node
from agents.base_agent import BaseAgent
from agents.skill_gen.skill_gen_agent_node import make_skill_gen_node
from agents.skill_gen.save_agent import make_save_agent_node
from llm.load_claude import load_llm
import json
import os

# Loading config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(BASE_DIR, "cfg", "config.yaml")
cfg = load_yaml_config(yaml_path)

# Loading prompt
data = load_prompt("skill_gen", cfg)

# Loading LLM
llm = load_llm()

# Create Langgraph nodes
graph = StateGraph(AgentState)
graph.add_node("skill_generator_agent", make_skill_gen_node(BaseAgent(llm), data, cfg))
graph.add_node("router", lambda state: state)
graph.add_node("save_agent", make_save_agent_node(cfg))

# Create Langgraph edges
graph.add_edge(START, "skill_generator_agent")
graph.add_edge("skill_generator_agent", "save_agent")
graph.add_edge("save_agent", "router")

# Create Langgraph conditional edges
graph.add_conditional_edges(
    "router",
    build_skill_router_node(cfg),
    {
        "continue": "skill_generator_agent",
        "end": END,
    }
)

#Compile graph
app = graph.compile()

with open("graph_skill_gen.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())


def print_stream(stream):
    for s in stream:
        messages = s.get("messages", [])
        if not messages:
            continue  # nothing to print

        message = messages[-1]

        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()



inputs = {
    "messages": [],
    "skills": [{}]
}

print_stream(app.stream(inputs, stream_mode="values"))


