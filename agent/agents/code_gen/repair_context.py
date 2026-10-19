"""Explicit current-attempt context; never replay stale code or thinking blocks."""
import json
from pathlib import Path
import re


def response_text(message):
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content
                     if isinstance(block, dict) and block.get("type") == "text")


def repair_prompt(template, state):
    values = {
        "CURRENT_SKILL_CODE": state.get("current_skill_code") or "No code was returned. Generate a complete implementation.",
        "SKILL_DESCRIPTION_JSON": json.dumps(state["skill_desc"]),
        "UNIT_TEST_FEEDBACK_JSON": state.get("evaluation_feedback") or "Not evaluated: execution must succeed first.",
        "EVALUATION_FEEDBACK_JSON": state.get("evaluation_feedback") or "Not evaluated: execution must succeed first.",
        "MOTION_FAILURE_DIAGNOSIS_FEEDBACK": state.get("diagnosis", ""),
        "SKILL_NAME": state["skill_name"],
    }
    # Single pass: braces inside supplied code/JSON are literal content.
    prompt = re.sub(r"\{(" + "|".join(values) + r")\}",
                    lambda match: values[match.group(1)], template)
    base = Path(__file__).resolve().parents[3] / "simulator/motion/base.py"
    return (prompt + "\n\nCURRENT EXECUTION FEEDBACK (fix this before trajectory quality):\n"
            + state.get("execution_feedback", "")
            + "\n\nCURRENT MOTION PLAN:\n" + state.get("motion_plan", "")
            + "\n\nAUTHORITATIVE SIMULATOR INTERFACE:\n"
            + "Import from motion.base import BaseMotionGenerator. Define one concrete "
            + "<SKILL_NAME>_MotionGenerator subclass. Implement "
            + "compute_foot_position_body_frame(self, leg_name, phase), returning a finite "
            + "numpy vector of shape (3,) for every leg. Use the provided leg names "
            + "(e.g. FL_calf), not guessed keys. Do not override step/reset or dynamically "
            + "install abstract methods. Do not invent substitute base classes.\n\n"
            + base.read_text())
