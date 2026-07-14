# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MimicAgent: an LLM agent (LangGraph + Claude) that turns a natural-language robot skill description into motion-generator code, executes it in a MuJoCo simulator, checks the resulting trajectory, and iteratively repairs the code. The top level has four areas:

- `robots/<name>/` — one self-contained directory per robot (spec, MuJoCo assets, reference skill)
- `simulator/` — the MuJoCo kinematic simulator that drives a robot with a skill and records a trajectory
- `agent/` — the LLM pipeline that writes skills (was `mimic_agent/`)
- `skill_output/<robot>/` — generated skills + artifacts (gitignored); `archive/` — old skills + input library

The whole system is morphology-agnostic: `robots/<name>/robot.yaml` is the **single source of truth** for a robot (end-effectors, joints + limits, passive joints, base-height envelope, camera, prompt "robot card"). Both halves read it; nothing else hardcodes morphology. Bundled robots: `unitree_go2` (legged quadruped), `unitree_go2w` (wheeled quadruped, 4 passive wheel joints), `humanoid` (28 DoF, 2 feet as IK targets).

## Commands

Run everything **from the repo root** (the simulator resolves data paths relative to the repo root; the agent runs its subprocess with the repo root as CWD).

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="..."

# Simulator: play a robot's reference skill (default mode = "example")
python simulator/run.py --robot unitree_go2 --record True
python simulator/run.py --robot humanoid --mode example --record True

# Simulator: play an agent-generated skill (from skill_output/<robot>/<skill>/)
python simulator/run.py --robot unitree_go2w --mode SKILL_TROT --record True

# Agent: single skill from natural language
cd agent && python run.py --skill "The robot trots forward" --name SKILL_TROT

# Agent: batch over the archived input library
cd agent && python run.py

# Interactive Streamlit demo
cd agent && streamlit run app.py
```

There is no test suite. `agent/utils/test_motion_quality.py` is not pytest — it is the trajectory-quality checker run by the pipeline's unit-test node.

Caveats for running:
- Robot MuJoCo models (MimicKit-style) are bundled under `robots/<name>/assets/`; each `scene.xml` adds a flat floor and a `home` keyframe. None define sites, so IK targets fall back to body positions (the startup warning is expected).
- The simulator opens a glfw window — it needs a display.
- **CWD must be the repo root.** `simulator/run.py` puts its own dir on `sys.path` (so `from core...`, `from utils...`, `from motion...` resolve), but data paths (`robots/...`, `skill_output/...`) are repo-root-relative. The agent selects the robot once in `agent/cfg/config.yaml` (`robot.name`) and passes `--robot` to the simulator subprocess, so the two halves can't disagree.

## Architecture

### Robot spec (`robots/<name>/robot.yaml` → `simulator/utils/robot_config.py:RobotSpec`)

Everything morphology-specific lives here: `end_effectors` (ordered `{name, body, ground_offset}` — the names are what skill code and prompts use, no normalization layer), `joints` (in **model qpos[7:] order** — the recorder and quality checks depend on this; `passive: true` marks unactuated joints like the go2w wheels), `base_height_range`, `camera`, `sim_freq`, `mjcf`, `example_skill` (path to the robot's reference skill), and `robot_card` (free-text morphology description injected into every prompt). Paths inside the yaml (`scene_xml`, `mjcf`, `example_skill`) are repo-root-relative. The controller asserts at startup that the YAML joint order matches the model's qpos order.

### Agent pipeline (`agent/`)

LangGraph state machine built in `agent/run.py` over `agents/code_gen/state.py:AgentState`:

```
skill_filter → motion_planner → code_generator → code_executer → unit_test → critic_router
                                      ↑                                          |
                                critic_coder ← motion_diagnosis ←────────── (loop)
```

- `code_executer_agent.py` is the bridge to the simulator: it extracts Python from the LLM message, writes it as `skill_output/<robot>/<SKILL>/<SKILL>_NNNN.py` (auto-incrementing), then runs the simulator via `subprocess` (`python simulator/run.py --mode <skill> --robot <robot> --record True`, cwd = repo root).
- `run_unit_test.py` loads the recorded eval pkl (`skill_output/<robot>/<skill>/<skill>.pkl`) and scores it with `agent/utils/test_motion_quality.py` (base-height envelope, ground penetration with per-EE `ground_offset`, joint limits, all-EE airtime — all parameterized by the RobotSpec); failures go to `motion_diagnosis_agent` → `critic_coder_agent`, which rewrites the code and re-executes. Loop ends on pass (`rewrite_mode == "none"`) or `critic.max_iters`.
- Everything a skill generates lives in one dir, `skill_output/<robot>/<skill>/`: code `<skill>_NNNN.py`, trajectory `<skill>.pkl` + `_mimickit.pkl`, `motion_plan_NNNN.json`, `report_NNNN.json`, `metadata.json`.
- Config: `agent/cfg/config.yaml` — LLM model, `robot.name`, four paths (`prompts_dir`, `robots_dir`, `simulator_dir`, `skill_output_dir`), `critic.max_iters`.
- Prompts: **one shared template set** in `agent/prompts/templates/*.txt` (`=== SYSTEM ===` / `=== HUMAN ===`). `prompts/prompt_loader.py:load_prompt` fills morphology placeholders from the RobotSpec via `str.replace` (never `.format` — templates contain literal JSON braces): `{robot}`, `{urdf}`, `{end_effectors}`, `{joint_order}`, `{joint_limits}`, `{base_height_range}`, `{morphology_card}`, `{example_skill}`, `{base_class_source}` (reads `simulator/motion/base.py` live, so the prompt always matches the real contract). Per-call placeholders (`{{MOTION_SPEC_INPUT}}`, `{SKILL_NAME}`, `{USER_SKILL}`, …) are filled later by the nodes.
- `agent/utils/robot_spec.py` loads `simulator/utils/robot_config.py` by file path (because `agent/utils` shadows the simulator's `utils` package).
- Input skill library (batch mode only) is archived at `archive/skill_library/skills_v1/all_skills.json`.

### Simulator (`simulator/`)

`simulator/run.py` → `simulator/core/`:
- `core/config.py` loads `simulator/config.yaml` (flattened into argparse-style args).
- `core/simulator.py` — MuJoCo + glfw rendering; camera framing comes from the spec.
- `core/controller.py:MotionControllerRunner` — loads the RobotSpec; if `mode == "example"` (the default) or the skill dir is absent, loads `robots/<robot>/example.py`, else loads `skill_output/<robot>/<mode>/`. Calls the skill's MotionGenerator for root pose + body-frame EE targets, solves IK (`utils/kinematics.py`, generic damped-least-squares over the spec's sites/joints with a null-space pull toward the standing pose, clamped to joint ranges each iteration), and applies the skill's optional `joint_overrides` (joints set directly instead of via IK — how a humanoid moves its arms/torso).
- `core/recorder.py` — writes two artifacts to `skill_output/<robot>/<mode>/`: `<mode>.pkl` (eval dict: `root_pos/root_quat/joints/ee_positions` + `fps/robot/ee_names`; consumed by the agent's unit-test node) and `<mode>_mimickit.pkl` (MimicKit `{fps, loop_mode, frames}`).

### Skills & the framework (`simulator/motion/`)

- `simulator/motion/__init__.py` — the loader. `get_motion_controller(skill, gaits_dir)` imports the **highest-numbered** `<skill>/<skill>_<N>.py` from `skill_output/<robot>/`; `get_example_controller(path)` loads a robot's single reference `example.py`. Both return the first concrete class whose name ends in `MotionGenerator`.
- All skills subclass `simulator/motion/base.py:BaseMotionGenerator`. Contract: `__init__(init_ee_pos: dict[name → body-frame pos], freq)`; implement `compute_ee_position_body_frame(ee_name, phase)`; optionally override `compute_joint_overrides(phase)` for non-IK joints. EE targets in **body frame**, base pose in **world frame**, phase ∈ [0,1]. This file's source is injected into the coder prompt — keep its docstrings accurate. Skills import `from motion.base import BaseMotionGenerator`.
- The tracked reference per robot is `robots/<name>/example.py`, which is what the default demo plays. Agent output in `skill_output/` is gitignored (regenerable). Pre-refactor go2w skills (old contract) are archived in `archive/skills_go2w_v1/`.

### Adding a robot

1. Create `robots/<name>/` with `robot.yaml` (end_effectors, joints in model qpos order with limits, base_height_range, camera, robot_card), `example.py` (subclass of `BaseMotionGenerator`), and `assets/` (MuJoCo model + a `scene.xml` with a floor + `home` keyframe).
2. Point `robot.yaml`'s `scene_xml` / `mjcf` / `example_skill` at the files under `robots/<name>/`.
3. Set `robot.name` in `agent/cfg/config.yaml`. No prompt or code changes needed. `python simulator/run.py --robot <name>` plays the example immediately.
