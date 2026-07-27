"""
prompts/prompt_loader.py

Loads the prompt templates (prompts/templates/*.txt) and fills the morphology
placeholders from the agent config.

Template format:
    === SYSTEM ===
    <system prompt>
    === HUMAN ===
    <human prompt>

Morphology placeholders (substituted here, via str.replace — templates contain
literal JSON braces, so .format() is never used):
    {robot}   robot name (cfg.robot.name)
    {urdf}    full MJCF/URDF text (cfg.robot.urdf)

Per-call placeholders — {{MOTION_SPEC_INPUT}}, {SKILL_DESCRIPTION_JSON},
{SKILL_NAME}, {USER_SKILL}, {CURRENT_SKILL_CODE}, {UNIT_TEST_FEEDBACK_JSON},
{EVALUATION_FEEDBACK_JSON}, {MOTION_FAILURE_DIAGNOSIS_FEEDBACK} — are left
untouched; the agent nodes fill those. {skills_per_generation} is likewise left
alone: agents/skill_gen/nodes.py fills it from cfg["dataset"] via .format().
"""

import os

import yaml


def load_yaml_config(yaml_path="config.yaml"):
    """Load the agent config, resolving relative paths against the repo root."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = os.path.dirname(os.path.dirname(os.path.abspath(yaml_path)))
    for key, value in cfg.get("paths", {}).items():
        if not os.path.isabs(value):
            cfg["paths"][key] = os.path.normpath(os.path.join(root, value))

    if "robot" in cfg and "urdf" in cfg["robot"]:
        urdf = cfg["robot"]["urdf"]
        if not os.path.isabs(urdf):
            cfg["robot"]["urdf"] = os.path.normpath(os.path.join(root, urdf))

    return cfg


def load_urdf_text(cfg):
    urdf_path = cfg["robot"]["urdf"]
    if os.path.isfile(urdf_path):
        with open(urdf_path, "r", encoding="utf-8") as f:
            return f.read()
    return f"<URDF/MJCF not found at {urdf_path}>"


def _build_substitutions(cfg):
    return {
        "{robot}": cfg["robot"]["name"],
        "{urdf}": load_urdf_text(cfg),
    }


def load_prompt(name, cfg):
    """Load prompts/templates/<name>.txt and fill the morphology placeholders."""
    prompt_path = os.path.join(cfg["paths"]["prompts_dir"], "templates", f"{name}.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "=== HUMAN ===" in content:
        system, human = content.split("=== HUMAN ===", 1)
    else:
        system, human = content, ""
    system = system.split("=== SYSTEM ===", 1)[1]

    data = {"system": system.strip(), "human": human.strip()}

    substitutions = _build_substitutions(cfg)
    for key in ("system", "human"):
        for placeholder, value in substitutions.items():
            data[key] = data[key].replace(placeholder, value)

    return data
