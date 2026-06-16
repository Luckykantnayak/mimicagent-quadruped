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