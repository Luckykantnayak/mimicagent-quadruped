from agents.code_gen.state import AgentState


def skill_router(state: AgentState) -> str:
    """
    Decides whether to proceed with code generation for the skill or END
    :param state:
    :return:
    """
    if state.get("success", True):
        return "continue"

    return "end"

