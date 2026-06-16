from agents.base_agent import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string
from agents.code_gen_v2.state import AgentState

from utils.tools import extract_json_from_message

import json


def make_skill_filter_node(agent: BaseAgent, data):
    def skill_filter(state: AgentState) -> str:
        """
        This function looks at the skill description and decides whether it's a feasible skill or not.
        Returns the edge 'continue' if skill is valid, else 'end' --> proceeds to next skill
        :param state:
        :return:
        """
        system_message = data["system"]  # str
        human_message = data["human"]  # str

        skill_description = json.dumps(state["skill_desc"])   # str
        human_message = human_message.replace('{SKILL_DESCRIPTION_JSON}', skill_description)

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]

        output = agent.generate(get_buffer_string(messages))  # output type: AIMessage

        output = extract_json_from_message(get_buffer_string([output]))  # JSON

        feasible = output["feasible"]

        if feasible:
            return "continue"

        return "end_skill"

    return skill_filter
