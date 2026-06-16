from agents.code_gen.state import AgentState

def make_critic_router(cfg):
    def critic_router(state: AgentState) -> str:
        if state["rewrite_mode"] == "none" or state["num_critic_iters"] > cfg["critic"]["max_iters"]:
            return "end_skill"

        else:
            return "loop"

    return critic_router