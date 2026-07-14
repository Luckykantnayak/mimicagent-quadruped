# MimicAgent — Architecture & Pipeline Guide

MimicAgent turns a **natural-language skill description** into a **verified motion
trajectory** for a robot. An LLM agent writes the code that generates the motion,
a MuJoCo simulator executes it kinematically and records a trajectory, a quality
checker scores it, and the agent repairs its own code until the motion passes.

The defining property of the system is that it is **morphology-agnostic**: a
single per-robot spec file drives both the simulator and the LLM prompts, so the
*same* pipeline produces skills for a legged quadruped, a wheeled quadruped, or a
28-DoF humanoid with no code or prompt changes.

---

## 1. The three pieces

```
                          robots/<name>/robot.yaml
                     (single source of truth: morphology)
                           /                        \
                          /                          \
              injected into prompts            read by the simulator
                        /                              \
        ┌──────────────────────────┐        ┌────────────────────────────┐
        │          agent/          │  runs  │        simulator/          │
        │  LLM pipeline that       │───────▶│  drives the robot with a   │
        │  WRITES a skill (code)   │◀───────│  skill, RECORDS a traj     │
        └──────────────────────────┘ traj   └────────────────────────────┘
                        │                              │
                        └──────── skill_output/<robot>/<skill>/ ─────────┘
                            (generated code + trajectory + reports)
```

- **`robots/<name>/`** — one self-contained directory per robot: the spec
  (`robot.yaml`), the MuJoCo model (`assets/`), and a curated reference skill
  (`example.py`).
- **`simulator/`** — the kinematic MuJoCo simulator. Given a robot and a skill,
  it plays the motion and records a trajectory.
- **`agent/`** — the LangGraph pipeline of LLM nodes that authors skills and
  drives the simulator in a self-correcting loop.

Generated artifacts land in **`skill_output/<robot>/<skill>/`** (git-ignored);
the input skill library and pre-refactor skills live in **`archive/`**.

### Repository layout

```
robots/
  unitree_go2/   unitree_go2w/   humanoid/
    robot.yaml        # spec: end-effectors, joints+limits, camera, robot card
    example.py        # reference skill (the default demo)
    assets/           # MuJoCo model: scene.xml + model.xml + meshes
simulator/
  run.py              # entry point (run from repo root)
  config.yaml         # run defaults (robot, mode, record settings)
  core/               # simulator.py  controller.py  recorder.py  config.py
  motion/             # base.py (BaseMotionGenerator) + loader __init__.py
  utils/              # kinematics.py  math_utils.py  robot_config.py  file_utils.py
agent/
  run.py              # code-gen pipeline entry (single skill or batch)
  run_skill_gen.py    # auto-mode: LLM invents a skill library
  app.py              # Streamlit demo
  agents/  cfg/  llm/  prompts/  utils/
skill_output/<robot>/<skill>/    # generated (git-ignored)
archive/                         # old skills + input skill library
```

---

## 2. Why it is morphology-agnostic

Everything a robot-specific decision could depend on is declared **once**, in
`robots/<name>/robot.yaml`, and consumed by both halves:

| What varies per robot | Where it's declared | Who uses it |
|---|---|---|
| End-effector names & bodies | `end_effectors` | skill code, prompts, IK, quality checks |
| Joints, order, limits | `joints` | recorder order, IK clamp, joint-limit check, prompt |
| Which joints are passive | `joints[].passive` | quality checks skip them (e.g. go2w wheels) |
| Contact geometry | `end_effectors[].ground_offset` | ground-penetration / airtime checks |
| Base-height envelope | `base_height_range` | base-height quality check |
| Camera framing | `camera` | renderer |
| The MJCF text | `mjcf` | injected into prompts as `{urdf}` |
| Free-text morphology notes | `robot_card` | injected into every prompt |
| Reference implementation | `example_skill` | injected into the coder prompt |

There is **no** `if humanoid:` branch anywhere. The three bundled robots differ
about as much as legged robots can, and all run through the identical pipeline:

| | `unitree_go2` | `unitree_go2w` | `humanoid` |
|---|---|---|---|
| End-effectors | 4 feet (FL/FR/RL/RR) | 4 wheels (FL/FR/RL/RR) | 2 feet (left/right) |
| Actuated joints | 12 | 12 | 28 |
| Passive joints | 0 | 4 (wheels) | 0 |
| Base body | `base` | `base` | `pelvis` |
| Base-height range | 0.10–0.55 m | 0.10–0.68 m | 0.35–1.60 m |
| Non-IK articulation | — | — | arms/torso/neck via `joint_overrides` |

The agent adapts because the prompt templates are filled from the spec (see §6);
the simulator adapts because the controller and IK read the spec directly.

---

## 3. The robot spec (`robots/<name>/robot.yaml`)

Loaded by `simulator/utils/robot_config.py:load_robot_spec` into a `RobotSpec`
dataclass. Annotated example (Go2W):

```yaml
name: unitree_go2w
scene_xml: robots/unitree_go2w/assets/scene.xml   # scene w/ floor + "home" keyframe
mjcf:      robots/unitree_go2w/assets/go2w_mimickit_style.xml   # injected as {urdf}
base_body: base                                   # free-joint body (camera + root)
sim_freq:  200

# Ordered end-effectors. `name` is what skill code & prompts use — no
# normalization layer. ground_offset = body-origin -> contact-point distance.
end_effectors:
  - {name: FL, body: FL_wheel_link, ground_offset: 0.1244}
  - {name: FR, body: FR_wheel_link, ground_offset: 0.1244}
  - {name: RL, body: RL_wheel_link, ground_offset: 0.1244}
  - {name: RR, body: RR_wheel_link, ground_offset: 0.1244}

# Joints in MODEL qpos[7:] ORDER (the recorder and checks depend on this).
# passive: true marks unactuated joints (excluded from IK & limit checks).
joints:
  - {name: FL_hip_joint,   limits: [-1.0472, 1.0472]}
  - {name: FL_thigh_joint, limits: [-1.5708, 3.4907]}
  - {name: FL_calf_joint,  limits: [-2.7227, -0.83776]}
  - {name: FL_wheel_joint, passive: true}
  # ... FR / RL / RR blocks ...

base_height_range: [0.10, 0.68]
camera: {lookat: [0.0, 0.0, 0.35], distance: 2.5, azimuth: 90, elevation: -20}

example_skill: robots/unitree_go2w/example.py

robot_card: |
  Free-text morphology description injected into every prompt. Tells the LLM
  what the end-effectors are, how limbs coordinate, and any rules (e.g. "wheel
  joints are passive — never command them").
```

`RobotSpec` exposes derived views used across the code: `ee_names`, `ee_bodies`,
`ee_ground_offsets`, `joint_names` (full, model order), `actuated_joint_names`,
`passive_joint_indices`, `joint_limits` (actuated only), plus `base_height_range`,
`camera`, `robot_card`, `example_skill`, `mjcf`.

**Invariant:** the `joints` list must match the model's `qpos[7:]` order. The
controller asserts this at startup, because the recorder stores the joint vector
in that order and the quality checks index into it.

---

## 4. The simulator — how a skill is executed

Entry: `python simulator/run.py --robot <name> --mode <skill> --record True`
(run from the repo root). Flow through `simulator/core/`:

```
run.py
  └─ load_robot_spec(robot)                     # robots/<robot>/robot.yaml
  └─ MujocoSimulator(scene_xml, camera)         # loads model, applies "home" keyframe
  └─ MotionControllerRunner(sim, args):
        1. build IK over spec.ee_bodies / spec.actuated_joint_names
        2. read the home base pose from qpos
        3. FK -> initial end-effector positions in the BODY frame  (init_ee_pos)
        4. load the skill:
             mode == "example"  -> robots/<robot>/example.py
             else               -> skill_output/<robot>/<mode>/<mode>_<N>.py
        5. instantiate  Motion(init_ee_pos=...)  and  reset(base_pose)
  └─ main loop, each step:
        state = runner.step():
          motion_state = skill.step(dt)          # root pose (world) + EE targets (body)
          EE targets -> world -> IK solve        # joint angles (clamped to limits)
          apply skill.joint_overrides            # non-IK joints (e.g. humanoid arms)
          write root pose + joints back to qpos
        recorder.record(state)                   # after a 50-frame warm-up
  └─ recorder.save() -> two pkls
```

### Initial conditions (every run)
- The robot starts at the **`home` keyframe** in `assets/scene.xml` (standing
  base height, nominal joint angles).
- The controller computes each end-effector's **body-frame position via forward
  kinematics** and hands it to the skill as `init_ee_pos`. **Skills author motion
  as deltas from these nominal positions.**
- The base starts at the keyframe pose and is driven **only by velocity
  integration in the world frame** (no absolute base trajectory).
- Recording skips the first **50 warm-up frames**, then captures
  `record_cycles / gait_freq × sim_freq` frames.

### Inverse kinematics (`simulator/utils/kinematics.py`)
Generic damped-least-squares over the spec's end-effector sites/bodies and
actuated joints, with a null-space pull toward the standing pose. It is N-EE
generic (4 wheels or 2 feet — same solver) and **clamps every joint to its model
range each iteration**, so recorded trajectories respect joint limits by
construction.

### Outputs (`simulator/core/recorder.py`)
Two files in `skill_output/<robot>/<skill>/`:
- `<skill>.pkl` — evaluation trajectory: `root_pos (T,3)`, `root_quat (T,4)`,
  `joints (T,J)`, `ee_positions (T,N,3)`, plus `fps`, `robot`, `ee_names`.
  Consumed by the agent's quality checker.
- `<skill>_mimickit.pkl` — MimicKit motion `{fps, loop_mode, frames}` where
  `frames = [root_pos(3), exp_map(3), joints(J)]`, for downstream imitation
  learning.

---

## 5. The skill contract (`simulator/motion/base.py`)

Every skill subclasses `BaseMotionGenerator`. The whole authoring surface:

```python
from motion.base import BaseMotionGenerator

class MySkill_MotionGenerator(BaseMotionGenerator):
    def __init__(self, init_ee_pos, freq=1.0):
        super().__init__(init_ee_pos, freq=freq)
        # self.base_ee_pos : dict{ee_name -> nominal body-frame position}
        # self.ee_names, self.freq, self.root_pos, self.root_quat

    # REQUIRED: body-frame target for one end-effector at a given phase
    def compute_ee_position_body_frame(self, ee_name, phase):
        pos = self.base_ee_pos[ee_name].copy()
        ...                       # perturb around the nominal position
        return pos                # np.ndarray (3,)

    # OPTIONAL: drive the base pose (default = integrate commanded velocity)
    def update_base_motion(self, phase, dt):
        ...                       # set self.root_pos / self.root_quat (WORLD frame)

    # OPTIONAL: set joints directly instead of via IK (non-IK articulation)
    def compute_joint_overrides(self, phase):
        return {"right_shoulder_x": angle, ...}   # {joint_name: radians}
```

**Frame conventions (hard rules, taught to the LLM):**
- End-effector targets are in the **body frame** (relative to the base).
- The base pose lives in the **world frame**.
- A single phase variable `phase ∈ [0,1]`, cyclic and smooth, drives everything.
- The base evolves **only** via velocity integration — no absolute base path.

`joint_overrides` is how a humanoid moves its arms/torso/neck (they are not IK
end-effectors); for the quadrupeds it is unused. `base.py`'s source is injected
verbatim into the coder prompt, so the contract the LLM sees always matches the
real one.

---

## 6. The agent pipeline (`agent/`)

A LangGraph state machine (`agent/run.py:build_graph`) with a self-repair loop:

```
        ┌───────────────┐   feasible?
START ─▶│ skill_filter  │───────────────▶ END (skip infeasible)
        └──────┬────────┘ yes
               ▼
        ┌───────────────┐   ┌────────────────┐   ┌────────────────┐
        │ motion_planner│──▶│ code_generator │──▶│ code_executer  │
        │  (JSON plan)  │   │  (Python code) │   │ run sim+record │
        └───────────────┘   └────────────────┘   └───────┬────────┘
                                                          ▼
                                                  ┌────────────────┐
                                                  │   unit_test    │
                                                  │ quality checks │
                                                  └───────┬────────┘
                                             pass │        │ fail
                                                  ▼        ▼
                                                 END   ┌──────────────────┐
                                                       │ motion_diagnosis │
                                                       │  (why it failed) │
                                                       └────────┬─────────┘
                                                                ▼
                                                       ┌──────────────────┐
                                                       │  critic_coder    │──▶ back to
                                                       │ (rewrite code)   │   code_executer
                                                       └──────────────────┘
              (loop bounded by critic.max_iters = 5)
```

**Node by node:**
- **skill_filter** — LLM judges whether the described skill is a realizable
  ground-robot behavior (rejects flying, teleporting, etc.) before any compute is
  spent.
- **motion_planner** — turns the description into a structured, phase-based plan
  (sub-phases, contact plan, coordination groups, base-motion intent) and sets the
  constraint flag `all_feet_off_ground_consistency` (true = grounded skill,
  false = aerial).
- **code_generator** — writes the `MotionGenerator` subclass, using the injected
  base-class source and the robot's `example.py` as the style reference.
- **code_executer** — writes `skill_output/<robot>/<skill>/<skill>_NNNN.py`, then
  runs `python simulator/run.py --mode <skill> --robot <robot> --record True`.
- **unit_test** — scores the recorded trajectory against the quality gates (§7).
  Pass ⇒ done. Fail ⇒ **motion_diagnosis** (explains the failure) ⇒
  **critic_coder** (rewrites, incrementing the file version) ⇒ re-execute.

### Prompts are one shared, spec-filled template set
`agent/prompts/templates/*.txt` (`coder`, `motion_planner`, `critic_coder`,
`motion_failure_diagnosis`, `skill_filter`, `skill_formatter`, `skill_gen`) —
`=== SYSTEM ===` / `=== HUMAN ===` sections. `prompts/prompt_loader.py` fills
morphology placeholders from the `RobotSpec` via `str.replace` (never `.format`,
because the templates contain literal JSON braces):

`{robot}` · `{urdf}` · `{end_effectors}` · `{num_end_effectors}` ·
`{joint_order}` · `{joint_limits}` · `{base_height_range}` · `{morphology_card}` ·
`{example_skill}` · `{base_class_source}` (read live from `simulator/motion/base.py`).

Per-call placeholders (`{{MOTION_SPEC_INPUT}}`, `{SKILL_NAME}`, `{USER_SKILL}`,
`{CURRENT_SKILL_CODE}`, `{skills_per_generation}`, …) are filled later by the
nodes. Adding a robot needs **no** prompt edits.

---

## 7. Constraints — what a generated skill must satisfy

Enforced by `agent/utils/test_motion_quality.py:evaluate_trajectory`, all
parameterized from the `RobotSpec`. A failure drives the critic repair loop.

| Check | Rule | Source of the threshold |
|---|---|---|
| `base_height_envelope` | base z stays inside the envelope | `robot.yaml: base_height_range` |
| `ground_penetration` | no EE contact point (`ee_z − ground_offset`) below the floor | `end_effectors[].ground_offset` |
| `joint_limits` | every actuated joint within range | `joints[].limits` |
| `all_ee_airtime` | *(skill-dependent)* unless aerial, not all EEs airborne past a threshold | plan flag `all_feet_off_ground_consistency` |

(Smoothness metrics — velocity/acceleration/jerk — are implemented but currently
disabled in the evaluator.)

**A skill is a data object** — the JSON the pipeline consumes:

```json
{
  "skill_id": "SKILL_TROT",
  "skill_name": "Steady Trot",
  "skill_intent": "Walk forward at a steady pace",
  "skill_description": "Phase 0-0.5: diagonal pair FL+RR swing ... Phase 0.5-1: ...",
  "think": "reasoning (stripped before code-gen)"
}
```

---

## 8. Generating skills

**Prerequisites:** `pip install -r requirements.txt` (needs
`langgraph` + `langchain-anthropic`), `export ANTHROPIC_API_KEY=...`, a display
for the glfw window, and **run from the repo root**. The target robot is set by
`robot.name` in `agent/cfg/config.yaml`.

### 8a. One skill from a prompt (single robot)

```bash
# 1. set agent/cfg/config.yaml -> robot.name: humanoid
cd agent
python run.py --skill "The robot performs a graceful ballet phrase" --name SKILL_BALLET
```

`--skill` accepts free text (an LLM formats it into the skill JSON) **or** a
ready-made skill JSON object. The result — code, trajectory, motion plan, and
quality report — lands in `skill_output/humanoid/SKILL_BALLET/`. Play it:

```bash
python simulator/run.py --robot humanoid --mode SKILL_BALLET --record True
```

### 8b. The same skill across all robots

There is no built-in "all robots" switch — loop and override `robot.name`:

```bash
for R in unitree_go2 unitree_go2w humanoid; do
  # point agent/cfg/config.yaml -> robot.name at $R (edit or sed), then:
  (cd agent && python run.py --skill "Trot forward steadily" --name SKILL_TROT)
done
```

Each robot gets its own morphology-adapted plan and code under
`skill_output/<robot>/`.

### 8c. Auto-mode — the LLM invents many skills, then generates them all

Two steps.

**Step A — the LLM writes a skill library** (`agent/run_skill_gen.py`). It calls
the model repeatedly, `skills_per_generation` (20) at a time, feeding back the
names already produced so it stays diverse, until it has `total_skills` (100).
Both numbers live in `agent/cfg/config.yaml → dataset`.

```bash
cd agent
python run_skill_gen.py          # writes skill_output/<timestamp>/SKILL_*.json
```

**Step B — run the library through the code-gen pipeline** (`agent/run.py` with
no `--skill`). It reads one `all_skills.json` and runs every skill through the
graph of §6.

```bash
cd agent
python run.py                    # batch mode
```

> **Wiring note:** Step A writes per-batch `SKILL_*.json` files under a timestamp
> directory; Step B's `load_all_skills` reads a single `all_skills.json`
> (currently pointed at `archive/skill_library/skills_v1/all_skills.json`). To
> batch-run a *freshly generated* library, concatenate the batches into one
> `{"skills":[...]}` file and point `paths.skill_library` at it. The archived
> library is a ready-made 100-skill set you can batch-run immediately.

### Interactive demo

```bash
cd agent && streamlit run app.py
```
Enter your API key, pick a robot, type a skill, and watch each node's output
stream live.

---

## 9. Adding a new robot or morphology

No code or prompt changes — just a new `robots/<name>/` directory.

1. **Create `robots/<name>/assets/`** with the MuJoCo model and a `scene.xml`
   that adds a flat floor and a `home` keyframe (base at standing height, nominal
   joint angles). Compute the standing base height and per-EE `ground_offset` from
   forward kinematics if you don't know them.

2. **Write `robots/<name>/robot.yaml`** (§3): `end_effectors` (ordered, with
   `ground_offset`), `joints` **in model qpos order** with `limits` (mark
   unactuated ones `passive: true`), `base_height_range`, `camera`, and a
   `robot_card` describing the morphology and any rules. Point `scene_xml`, `mjcf`,
   and `example_skill` at the files under `robots/<name>/`.

3. **Write `robots/<name>/example.py`** — a hand-authored reference skill
   (subclass of `BaseMotionGenerator`). This is the default demo *and* the style
   reference injected into the coder prompt, so make it clean.

4. **Point the agent at it**: set `robot.name: <name>` in `agent/cfg/config.yaml`.

5. **Verify**: `python simulator/run.py --robot <name>` plays the example
   immediately; then generate a skill as in §8.

If the new robot has non-IK articulation (arms, a tail, a gripper), express it
through `compute_joint_overrides` — no new machinery required. The IK, recorder,
and all four quality checks already generalize to any end-effector count and any
joint set from the spec.

---

## 10. Worked example — `SKILL_BALLET` (humanoid)

A ballet phrase generated for the 28-DoF humanoid, demonstrating the
morphology-agnostic contract and non-IK articulation. One phase cycle:

- **Plié → relevé** — the pelvis breathes down then up (0.82→0.94 m) via
  `update_base_motion`. Each supporting foot is compensated in the body frame so
  it stays planted on the floor in the world frame (knees bend, legs extend).
- **Port de bras** — the arms sweep *bras bas* (low) → *à la seconde* (out) →
  *en haut* (overhead) and back, elbows rounding, via `compute_joint_overrides`
  on the shoulder/elbow joints (which are **not** IK end-effectors).
- **Promenade turn** — a full 360° eased pirouette in place (base yaw).
- **Dégagés** — the feet (the only IK end-effectors) point front/side one at a
  time, so at least one foot always supports.

It passed all four quality gates on the first attempt — feet planted (contact
height 0.001 m), base inside the envelope, joints in range, never fully airborne.
The exact same base class, IK, recorder, and checker that run the quadruped
walks produced this humanoid ballet with no special-casing. See
`skill_output/humanoid/SKILL_BALLET/` after running:

```bash
python simulator/run.py --robot humanoid --mode SKILL_BALLET --record True
```

---

## 11. Quick reference

| Task | Command (from repo root) |
|---|---|
| Play a robot's reference skill | `python simulator/run.py --robot <name> --record True` |
| Play a generated skill | `python simulator/run.py --robot <name> --mode <SKILL> --record True` |
| Generate one skill from a prompt | `cd agent && python run.py --skill "..." --name <SKILL>` |
| Auto-invent a skill library | `cd agent && python run_skill_gen.py` |
| Batch-generate a library | `cd agent && python run.py` |
| Interactive demo | `cd agent && streamlit run app.py` |
| Add a robot | new `robots/<name>/` (spec + `example.py` + `assets/`); set `robot.name` |

Key files: `robots/<name>/robot.yaml` (morphology), `simulator/motion/base.py`
(skill contract), `simulator/core/controller.py` (execution),
`agent/run.py` (pipeline), `agent/prompts/templates/` (prompts),
`agent/utils/test_motion_quality.py` (quality gates).
