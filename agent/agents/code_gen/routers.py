from agents.code_gen.state import AgentState


def make_unit_test_router(cfg):
    def unit_test_router(state: AgentState) -> str:
        """
        After code execution:
          - ran cleanly  -> run the quality checks ('continue')
          - repeated IDENTICAL pre-sim lint hard-fail -> REPLAN once (the fault is a
            structural PLAN bug the coder keeps patching around — e.g. a foot never
            swung to where a later phase declares it planted; a code rewrite can't fix
            a schedule the plan got wrong).
          - crashed / other failure -> repair the code ('repair'), capped by max_iters
          - out of iters -> give up ('end_skill')
        """
        if state.get("success", False):
            return "continue"
        if state.get("num_critic_iters", 0) > cfg["critic"]["max_iters"]:
            return "end_skill"
        if (state.get("lint_fail_streak", 0) or 0) >= 2 and not state.get("replanned", False):
            return "replan"
        return "repair"

    return unit_test_router


def make_critic_router(cfg):
    def critic_router(state: AgentState) -> str:
        if state["rewrite_mode"] == "none" or state["num_critic_iters"] > cfg["critic"]["max_iters"]:
            return "end_skill"

        # RE-PLAN (once): if code rewrites have stopped improving the score for a
        # couple of iterations, the fault is likely in the PLAN — a wrong control
        # channel a code edit can't fix, OR (now that fidelity drives the loop) a
        # fidelity invariant that is itself wrong for the motion the coder chose.
        # Step back to the planner to re-decide the interface / re-declare the
        # invariant with the readout in hand — the "step back and re-decide" a human
        # does. Fires whether the block is a floor failure or a fidelity failure;
        # bounded to once, still inside the num_critic_iters budget.
        # `stuck` (set in run_unit_test) is the SINGLE source of truth both this router
        # and the planner read — the score plateaued for >=2 iters OR a core test has
        # been stuck failing for >=3. Step back to the planner (once) to re-decide the
        # interface / a frozen-out base decision a code edit can't reach.
        if state.get("stuck", False) and not state.get("replanned", False):
            return "replan"

        return "loop"

    return critic_router
