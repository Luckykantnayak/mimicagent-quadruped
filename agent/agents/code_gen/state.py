from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence
from typing_extensions import TypedDict, Dict, Any
# from pydantic import BaseModel, Field

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    skill_name: str
    skill_version: str
    success: bool
    num_critic_iters: int
    rewrite_mode: str
    skill_desc: Dict[str, Any]
    token_usage: Dict[str, Any]
    current_skill_code: str
    motion_plan: str
    execution_feedback: str
    evaluation_feedback: str
    diagnosis: str
