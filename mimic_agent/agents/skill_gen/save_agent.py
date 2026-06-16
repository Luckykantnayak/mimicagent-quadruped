import re
from agents.skill_gen.state import AgentState
from langchain_core.messages import get_buffer_string

import json
import os
from datetime import datetime


def make_save_agent_node(cfg):
    # Create run directory ONCE
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = cfg["paths"]["skill_output_dir"]
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    def save_agent(state: AgentState):
        save(state, run_dir)
        return state

    return save_agent


def save(state: AgentState, run_dir):
    raw_text = get_buffer_string([state['messages'][-1]])

    # Extract clean JSON object
    json_data = extract_json_from_message(raw_text)

    # Find existing SKILL_*.json files
    existing = os.listdir(run_dir)
    nums = []

    for fname in existing:
        match = re.match(r"SKILL_(\d+)\.json", fname)
        if match:
            nums.append(int(match.group(1)))

    next_num = max(nums) + 1 if nums else 1

    file_path = os.path.join(run_dir, f"SKILL_{next_num:04d}.json")
    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=2)


def extract_json_from_message(text: str):
    """
    Extract JSON from an LLM message that may contain:
    - 'AI:' prefix
    - ```json ... ``` fences
    """
    # Remove leading "AI:" if present
    text = re.sub(r"^\s*AI:\s*", "", text)

    # Extract fenced JSON if present
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # Strip whitespace
    text = text.strip()

    # Validate / parse JSON
    return json.loads(text)