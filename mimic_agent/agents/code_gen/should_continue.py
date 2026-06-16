from agents.code_gen.state import AgentState


def should_continue(state: AgentState) -> str:
    """
    Determines whether to proceed ahead with unit tests or not
    :param state:
    :return:
    """
    if state.get("success", False):
        return "continue"

    else:
        return "end_skill"