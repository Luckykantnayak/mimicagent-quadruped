# agents/base_agent.py
from langchain_core.messages import AIMessage

class BaseAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, messages) -> AIMessage:
        return self.llm.invoke(messages)
