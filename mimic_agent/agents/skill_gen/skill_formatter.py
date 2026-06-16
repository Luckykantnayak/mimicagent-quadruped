import json

from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string

from agents.base_agent import BaseAgent
from utils.tools import extract_json_from_message


def is_structured_skill(raw):
    """True if `raw` is already a skill object (JSON dict with a skill_description)."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(obj, dict) and "skill_description" in obj


def format_user_skill(agent: BaseAgent, data, description, name=None):
    human_message = data["human"].replace("{USER_SKILL}", description)

    messages = [
        SystemMessage(content=data["system"]),
        HumanMessage(content=human_message),
    ]

    output = agent.generate(get_buffer_string(messages))
    skill = extract_json_from_message(get_buffer_string([output]))

    if name:
        skill["skill_id"] = name

    return skill


def prepare_skill(agent: BaseAgent, data, raw, name=None):
    """Return a skill dict for the pipeline.

    If `raw` is already a structured skill, it is used as-is (only filling a
    skill_id/skill_name if missing); otherwise it is treated as a natural-language
    request and converted by the formatter agent.
    """
    if is_structured_skill(raw):
        skill = json.loads(raw)
        if name:
            skill["skill_id"] = name
        skill.setdefault("skill_id", name or "SKILL_CUSTOM")
        skill.setdefault("skill_name", skill["skill_id"])
        return skill

    return format_user_skill(agent, data, raw, name=name)
