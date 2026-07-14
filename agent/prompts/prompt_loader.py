"""
prompts/prompt_loader.py

Loads the shared prompt templates (prompts/templates/*.txt) and fills the
morphology placeholders from the robot spec (robots/<name>.yaml at repo root).

Template format:
    === SYSTEM ===
    <system prompt>
    === HUMAN ===
    <human prompt>

Morphology placeholders (substituted here, via str.replace — templates
contain literal JSON braces, so .format() is never used):
    {robot}              robot name
    {urdf}               full MJCF/URDF text
    {end_effectors}      comma-separated end-effector names
    {num_end_effectors}  end-effector count
    {joint_order}        comma-separated actuated joint names, in order
    {joint_limits}       one "name: [lo, hi]" line per actuated joint
    {base_height_range}  e.g. "0.10 m to 0.68 m"
    {morphology_card}    free-text robot card from the spec
    {example_skill}      reference skill implementation source
    {base_class_source}  gaits/base.py source (the contract skills must follow)

Per-call placeholders like {{MOTION_SPEC_INPUT}}, {SKILL_NAME},
{CURRENT_SKILL_CODE} are left untouched — the agent nodes fill those.
"""

import os
import yaml

from utils.robot_spec import load_robot_spec


def load_yaml_config(yaml_path="config.yaml"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = os.path.dirname(os.path.dirname(os.path.abspath(yaml_path)))
    for key, value in cfg.get("paths", {}).items():
        if not os.path.isabs(value):
            cfg["paths"][key] = os.path.normpath(os.path.join(root, value))

    return cfg


def _read_text(path, what):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return f"<{what} not found at {path}>"


def _build_substitutions(cfg):
    spec = load_robot_spec(cfg)
    repo_root = cfg["paths"]["simulator_dir"]

    joint_limits = "\n".join(
        f"{name}: [{lo}, {hi}]" for name, (lo, hi) in spec.joint_limits.items()
    )
    lo, hi = spec.base_height_range

    return {
        "{robot}": spec.name,
        "{urdf}": _read_text(os.path.join(repo_root, spec.mjcf), "URDF/MJCF"),
        "{end_effectors}": ", ".join(spec.ee_names),
        "{num_end_effectors}": str(len(spec.ee_names)),
        "{joint_order}": ", ".join(spec.actuated_joint_names),
        "{joint_limits}": joint_limits,
        "{base_height_range}": f"{lo} m to {hi} m",
        "{morphology_card}": spec.robot_card.strip(),
        "{example_skill}": _read_text(
            os.path.join(repo_root, spec.example_skill), "example skill"
        ),
        "{base_class_source}": _read_text(
            os.path.join(repo_root, "simulator", "motion", "base.py"), "base class"
        ),
    }


def load_prompt(name, cfg):
    """Load prompts/templates/<name>.txt and fill morphology placeholders."""
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
