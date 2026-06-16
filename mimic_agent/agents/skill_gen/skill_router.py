from agents.skill_gen.state import AgentState

def build_skill_router_node(cfg):
    def skill_router(state: AgentState) -> str:
        max_iter = cfg["dataset"]["total_skills"]
        if len(state["skills"]) <= max_iter:
            return "continue"

        return "end"

    return skill_router
