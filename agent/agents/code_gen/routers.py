from agents.code_gen.state import AgentState


def make_should_continue(cfg):
    """
    After code execution:
      - it ran            -> score the trajectory ("continue")
      - it crashed        -> feed the traceback to the critic ("repair"),
                             bounded by critic.max_iters
      - out of iterations -> give up ("end_skill")
    """
    def should_continue(state: AgentState) -> str:
        if state.get("success", False):
            return "continue"

        if state.get("num_critic_iters", 0) <= cfg["critic"]["max_iters"]:
            return "repair"

        return "end_skill"

    return should_continue


def make_critic_router(cfg):
    def critic_router(state: AgentState) -> str:
        if state["rewrite_mode"] == "none" or state["num_critic_iters"] > cfg["critic"]["max_iters"]:
            return "end_skill"

        else:
            return "loop"

    return critic_router
