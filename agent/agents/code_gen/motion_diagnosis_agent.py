import os

from agents.base_agent import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string
from agents.code_gen.state import AgentState
from utils.read_latest_version import get_latest_json


def _load_plan(state, cfg):
    """The motion plan the code was built from — its per-limb channel assignments
    let the diagnosis judge whether a channel is wrong for what a limb must do."""
    try:
        skill_dir = os.path.join(
            cfg["paths"]["skill_output_dir"], cfg["robot"]["name"], state["skill_name"])
        with open(get_latest_json(skill_dir, "motion_plan")) as f:
            return f.read()
    except Exception:
        return "(motion plan unavailable)"


def make_motion_diagnosis_agent_node(coder_agent: BaseAgent, data, cfg):
    def node(state: AgentState) -> AgentState:
        system_message = data["system"]  # str
        human_message = data["human"].replace("{MOTION_PLAN_JSON}", _load_plan(state, cfg))
        human_message = human_message + "\n\n Context: " + get_buffer_string(state["messages"]) + "\n\n" + \
                        ("Ensure to review the previous diagnosis reports and rewards and suggest best. If a previous"
                         "iteration led to better rewards then take positives from there. Otherwise, take negatives from there"
                         "There can be many causes of joint limit violations (rapid motions/ body sinking/ etc ). "
                         "If a failure has persisted across iterations WITHOUT its metric shrinking, suspect an "
                         "INTERFACE MISMATCH (a limb on the wrong control channel) and name the correct channel.")

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]

        output = coder_agent.generate(get_buffer_string(messages))  # output type: AIMessage
        # Get Token Usage
        input_tokens = output.response_metadata['usage']['input_tokens']
        output_tokens = output.response_metadata['usage']['output_tokens']

        if state["token_usage"]:
            token_dict = {
                "input_tokens": input_tokens + state["token_usage"]["input_tokens"],
                "output_tokens": output_tokens + state["token_usage"]["output_tokens"],
            }

        else:
            token_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens}


        return {
            "messages": [output],
            'skill_name': state['skill_name'],
            'skill_version': state['skill_version'],
            'success': state['success'],
            "num_critic_iters": state['num_critic_iters'],
            'rewrite_mode': state['rewrite_mode'],
            'skill_desc': state['skill_desc'],
            'token_usage': token_dict,
        }

    return node
