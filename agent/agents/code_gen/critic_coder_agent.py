from agents.base_agent import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string
from agents.code_gen.state import AgentState
from agents.code_gen.repair_context import repair_prompt


def make_critic_coder_node(coder_agent: BaseAgent, data, cfg):
    def node(state: AgentState) -> AgentState:
        system_message = data["system"]  # str
        human_message = repair_prompt(data["human"], state)

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
