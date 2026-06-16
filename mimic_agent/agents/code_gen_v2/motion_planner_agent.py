from agents.base_agent import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string
from agents.code_gen_v2.state import AgentState
from utils.tools import extract_json_from_message

import os
import json
import re

def make_motion_plan_node(motion_planner_agent: BaseAgent, data, cfg):
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

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=context_prompt),
            HumanMessage(content=human_message),
        ]

        output = motion_planner_agent.generate(get_buffer_string(messages))  # output type: AIMessage
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

        save(skill_name=state["skill_name"],
             skill_version=state["skill_version"],
             data=extract_json_from_message(get_buffer_string([output])),
             cfg=cfg
             )

        return {
            "messages": [output],
            'skill_name': state['skill_name'],
            'skill_version': state['skill_version'],
            'success': state['success'],
            "num_critic_iters": state['num_critic_iters'],
            'rewrite_mode': state['rewrite_mode'],
            'skill_desc': state['skill_desc'],
            "token_usage": token_dict,

        }

    return node


def save(skill_name: str, skill_version: str, data, cfg):

    if not os.path.exists(os.path.join(cfg["paths"]["motion_planner_output_dir"], skill_version, skill_name)):
        os.makedirs(os.path.join(cfg["paths"]["motion_planner_output_dir"], skill_version, skill_name))

    existing = os.listdir(os.path.join(cfg["paths"]["motion_planner_output_dir"], skill_version, skill_name))
    nums = []

    for fname in existing:
        match = re.match(rf"{skill_name}_(\d+)\.json", fname)
        if match:
            nums.append(int(match.group(1)))

    next_num = max(nums) + 1 if nums else 1

    file_pth = os.path.join(cfg["paths"]["motion_planner_output_dir"], skill_version, skill_name, f"{skill_name}_{next_num:04d}.json")

    with open(file_pth, 'w') as f:
        json.dump(data, f, indent=4)

    return