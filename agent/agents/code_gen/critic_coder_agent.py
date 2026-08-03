from agents.base_agent import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string
from agents.code_gen.state import AgentState


def make_critic_coder_node(coder_agent: BaseAgent, data, cfg):
    def node(state: AgentState) -> AgentState:
        system_message = data["system"]  # str
        human_message = data["human"]  # str
        human_message += "\n\n Context: \n" + get_buffer_string(state["messages"]) + "\n\n" + \
                        ("Ensure to review the previous diagnosis reports and code and reward. If a previous"
                         "iteration led to better rewards then take positives from there. Otherwise, take negatives from there"
                         "There can be many causes of joint limit violations (rapid motions/ body sinking/ etc)"
                         "Also, if you are using any helper functions that are already not there (e.g., 'transform_body_to_world'), please write the helper functions yourself ")

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
            "last_prompt": get_buffer_string(messages),   # exact prompt sent (saved per version)
            'skill_name': state['skill_name'],
            'skill_version': state['skill_version'],
            'success': state['success'],
            "num_critic_iters": state['num_critic_iters'],
            'rewrite_mode': state['rewrite_mode'],
            'skill_desc': state['skill_desc'],
            'token_usage': token_dict,
        }

    return node