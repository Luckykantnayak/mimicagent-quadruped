# Adding a Robot

How to bring a new humanoid (G1, H1, your own) into MimicAgent.

Adding a robot is **assets + one spec file + one reference skill**. No prompt edits,
no code changes: `agent/prompts/templates/*.txt` contain no robot-specific text at
all, and every morphology fact reaches both the simulator and the LLM through
`robots/<name>/robot.yaml`.

**Contents**
1. [What you need](#1-what-you-need)
2. [MJCF requirements](#2-mjcf-requirements)
3. [Scaffold the spec](#3-scaffold-the-spec)
4. [`robot.yaml` field reference](#4-robotyaml-field-reference)
5. [Write `example.py`](#5-write-examplepy)
6. [Generate the physical summary](#6-generate-the-physical-summary)
7. [Point the agent at it](#7-point-the-agent-at-it)
8. [Verification checklist](#8-verification-checklist)
9. [Constants that are scale-dependent](#9-constants-that-are-scale-dependent)
10. [Scenes and environments](#10-scenes-and-environments)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What you need

| item | where | derivable? |
|---|---|---|
| MuJoCo model + meshes | `assets/robots/<name>/` | you supply |
| `robot.yaml` | `robots/<name>/` | **scaffolded** from the MJCF, then reviewed |
| `example.py` | `robots/<name>/` | **hand-written** (a short reference walk) |
| `summary.md`, `dynamics.json` | `robots/<name>/` | **generated** |
| `environments:`, `exemplars:` | in `robot.yaml` | optional; hand-authored |

Two of these are genuinely manual: the `# REVIEW` lines in the scaffolded spec, and
`example.py`. Everything else is a command.

---

## 2. MJCF requirements

The scaffolder and the physics extraction read the model directly. Following these
conventions is what makes the rest automatic.

| requirement | why |
|---|---|
| Root body carries a `<freejoint>` | the base pose is `qpos[0:7]`; actuated joints are `qpos[7:]` |
| Each end-effector body has `<site name="ee_<name>"/>` | marks it as an EE for auto-detection **and** gives IK a real target. No site ⇒ IK aims at the body origin |
| Each actuated joint has `range` | position limits, used by IK clamping and the `joint_limits` check |
| Each actuated joint has `actuatorfrcrange` | torque, used to derive velocity/acceleration ceilings |
| Passive joints have no actuator | marked `passive: true` in the spec |
| A `<keyframe name="home">` | the rest pose; also the IK null-space reference |
| Mass/inertia via `<inertial>` or geom density | needed for the dynamics tier |
| `meshdir="../robots/<name>/assets/"` | scenes `<include>` the robot from `assets/scenes/`, so mesh paths resolve relative to that wrapper |

### ⚠️ The home keyframe must not put a joint at its limit

This is the single most important modelling rule, and it is easy to violate.

If the `home` pose leaves a joint **exactly at the end of its range with the limb
straight**, that configuration is an *exact kinematic singularity* — the joint's
Jacobian column has zero first-order authority over the direction the limb needs to
move. A damped least-squares solver started there returns **exactly zero** correction
and cannot escape.

Concretely, on the bundled humanoid the knees sat at exactly `0.0` with range
`[0.0, 2.79]` and the legs straight. Lowering the base 0.88 → 0.36 m with the feet
held still left the knee at `0.000` on *every* frame and the foot **0.47 m
underground** — a 0.52 m tracking error. That is the squat/crouch/landing primitive
underneath every scene-interaction and parkour skill, so it silently broke box-carry,
ledge-drop and stair-climbing at once.

The framework now defends against this (`simulator/utils/kinematics.py` nudges any
at-limit joint a small margin inside its range before the first Jacobian), so a model
that violates the rule still works. But **give the home pose a few degrees of knee and
elbow flexion anyway** — no real robot stands with locked joints, and a
singularity-free rest pose gives the solver a much better warm start.

---

## 3. Scaffold the spec

```bash
python simulator/tools/gen_robot_spec.py --robot <name>
```

Writes `robots/<name>/robot.yaml`.

**Exact from the MJCF** (trust these): `joints` in `qpos[7:]` order with limits,
`base_body`, passive flags, paths, a first-draft `robot_card`.

**Marked `# REVIEW`** (check every one):

| field | what the guess is | why it matters |
|---|---|---|
| `end_effectors` | from `ee_*` sites, else a leaf-body heuristic | wrong set ⇒ IK drives the wrong limbs |
| `ground_offset` | estimated from geometry | **load-bearing** — every contact law is `surface_height + ground_offset` |
| `base_height_range` | from the home base z | too tight ⇒ legal crouches/jumps rejected |
| `camera` | a default framing | cosmetic, affects recorded video only |

---

## 4. `robot.yaml` field reference

### Top level

```yaml
name: humanoid                                  # must match the directory name
kind: humanoid                                  # free-text label
mjcf: assets/robots/humanoid/humanoid.xml       # repo-root-relative
scene_xml: assets/scenes/flat.xml               # default scene
base_body: pelvis                               # the body carrying the freejoint
sim_freq: 200                                   # Hz
null_space_gain: 0.02                           # see below
base_height_range: [0.30, 2.20]                 # metres, generous
camera: {lookat: [0.0, 0.0, 0.9], distance: 4.0, azimuth: 90, elevation: -15}
example_skill: robots/humanoid/example.py
```

`null_space_gain` is how hard IK pulls unconstrained DoFs back toward the rest pose.
**Lower it on a high-DoF robot** so limbs can reach far — 0.02 for this 28-DoF
humanoid. Too high and the arms refuse to extend; too low and redundant joints drift.

### `end_effectors` — ordered, IK-driven

```yaml
end_effectors:
  - {name: right_foot, body: right_foot, site: ee_right_foot, ground_offset: 0.05}
  - {name: right_hand, body: right_hand, site: ee_right_hand, contact: false, ground_offset: 0.04}
```

| key | meaning |
|---|---|
| `name` | **what skill code and prompts use.** There is no normalization layer — pick names a language model will use naturally (`right_foot`, not `rf_link`) |
| `body` | the MJCF body |
| `site` | the `ee_*` site; omit and IK aims at the body origin |
| `ground_offset` | body origin → contact point (sole/palm). Get this right |
| `contact: false` | IK-driven but **not** a ground contact — excluded from ground/penetration checks. Use for hands |
| `wheel: true` | a planted contact that may roll without being XY-pinned |

### `contact_bodies` — the rest of the weight-bearing roster

```yaml
contact_bodies:
  - {name: pelvis, body: pelvis, ground_offset: 0.09}   # buttocks (SIT)
  - {name: right_knee, body: right_shin, ground_offset: 0.06}   # kneel / crawl
```

Same shape, no `site`. **`end_effectors` + `contact_bodies` = the union IK roster**:
give one a world target and IK places it like a foot; return `None` and it is FK-only.

**Be generous here.** This roster is what makes the robot versatile — a sit needs the
pelvis, a plank the forearms, a crawl the shins, a headstand the head. Anything you
omit cannot be declared as a contact, so those skills become unbuildable.

### `joints` — in model `qpos[7:]` order

```yaml
joints:
  - {name: abdomen_x, limits: [-1.0472, 1.0472]}
  - {name: left_wheel, limits: [-inf, inf], passive: true}
```

**The order is not cosmetic** — the recorder and every quality check index into it.
The controller asserts at startup that it matches the model, so a mismatch fails loudly.

### `robot_card` — free text injected into every prompt

**Kinematic facts only**: DoF count, end-effectors, nominal base height, and each
joint's axis + sign + range. Deliberately says *nothing* about which control channel a
limb should use — that is the agent's decision, made per limb per phase. Adding
pose cookbooks or heuristics here is how you make the framework worse, not better.

### `environments` — optional named scenes

```yaml
environments:
  - name: chair
    scene: assets/scenes/chair.xml
    grounding: "off"        # QUOTE IT — unquoted `off` is YAML boolean False
    objects: []
    description: "A chair ahead: a SEAT panel and a BACKREST, both SOLID to the floor."
```

`grounding` picks the contact layer: `contact` (register base + pin + clamp) for
flat-floor skills; `"off"` for skills that author an absolute base and world-frame
contacts on raised surfaces; `pin` for pin+clamp only.

### `exemplars` — optional curated reference skills

```yaml
exemplars:
  - name: SKILL_CLIMB
    environment: ledge
    tags: [climb, mantle]
    summary: "Climbs onto the slab using hands then feet."
```

Backed by `robots/<name>/exemplars/<NAME>.py`. The coder is shown the **full source**
of an exemplar whose environment matches the one it picked, so these are high-leverage:
one good exemplar per scene topology teaches the pattern.

---

## 5. Write `example.py`

The reference skill — a short walk is fine. It is the default demo *and* is injected
into every prompt as `{example_skill}`, so its quality propagates.

Subclass `simulator/motion/base.py:BaseMotionGenerator`. Read that file — its
docstrings are the contract and are injected into the coder prompt verbatim.

```python
from motion.base import BaseMotionGenerator

class WalkMotionGenerator(BaseMotionGenerator):        # name must end in MotionGenerator
    def __init__(self, init_ee_pos, freq):
        super().__init__(init_ee_pos, freq)

    def compute_ee_position_world_frame(self, ee_name, phase): ...   # world-fixed target, or None
    def compute_ee_position_body_frame(self, ee_name, phase): ...    # torso-relative target, or None
    def compute_joint_overrides(self, phase): ...                    # {joint: angle}
    def contacts(self, phase): ...                                   # ["right_foot", ...]; [] = flight
```

Three control channels, chosen **per limb per phase**:

| channel | use when |
|---|---|
| world-frame IK | the world fixes the endpoint — a contact, or a limb gripping an object |
| body-frame IK | the target is relative to the torso — a swing foot |
| joint overrides | the limb's *shape* is the point and its endpoint is free — a gesture, a kick |

Both IK channels default to `None`, which makes IK skip that EE. `contacts(phase)`
returning `[]` declares ballistic flight.

---

## 6. Generate the physical summary

```bash
python simulator/tools/gen_robot_summary.py --robot <name>
```

Produces:
- **`summary.md`** — limb lengths, per-EE reach, per-joint limits, and (when the MJCF
  has mass + torque) velocity/acceleration ceilings and grounded speed/jump estimates.
  Injected as `{robot_summary}` so the agent commands feasible amplitudes.
- **`dynamics.json`** — machine-readable per-joint ceilings plus the `reach` block that
  powers the planner's reach law and the runtime slew limit.

The dynamics tier is automatic: emitted when the MJCF carries mass/inertia and a torque
source, skipped otherwise (the robot still runs, those checks just self-disable).

---

## 7. Point the agent at it

```yaml
# agent/cfg/config.yaml
robot:
  name: <name>
```

Or per-process, so two robots can generate in parallel:

```bash
ROBOT_NAME=<name> python agent/run.py --skill "..." --name SKILL_X
```

---

## 8. Verification checklist

Run these in order; each catches a different class of mistake.

```bash
# 1. The model loads, the spec matches it, the example plays.
python simulator/run.py --robot <name> --record True
```
Expect `[IK] Sites found: N/M`. `⚠️ Site not found` for `contact_bodies` is normal —
they usually declare no site, so IK aims at the body origin.

```bash
# 2. The spec's joint order matches the model.
```
The controller asserts this at startup — a silent start means it passed.

```bash
# 3. Reach and geometry sanity on the example.
python simulator/tools/reach_lint.py --mode example --robot <name> --grounding off
```
Pass `--scene` explicitly when linting by hand — a generated skill otherwise falls back
to `flat.xml` and every surface answer is wrong.

```bash
# 4. The example scores clean.
python agent/score_skills.py
```

```bash
# 5. End-to-end: one generated skill.
cd agent && python run.py --skill "The robot walks forward" --name SKILL_WALK
```

---

## 9. Constants that are scale-dependent

Most thresholds already scale to the robot — `self_collision` derives from the recorded
limb radius, grip tolerances from `2.5 × limb_radius + the object's half-extent`.

These do **not**. They are absolute metres, tuned for a ~1.6 m humanoid:

| constant | file | value | meaning |
|---|---|---|---|
| `NO_SURFACE_M` | `simulator/tools/reach_lint.py` | 0.25 | a declared contact hovering more than this has nothing under it |
| `ARRIVAL_JUMP_M` | `simulator/tools/reach_lint.py` | 0.25 | an authored contact target jumping this far in one frame teleported into contact |
| `SLIDE_GROSS_M` | `simulator/tools/reach_lint.py` | 0.25 | authored drag within one contact run (opt-in, `NO_DRAG=1`) |
| `REACH_SEAT_CAP` | `simulator/core/controller.py` | 0.15 | hard cap on base re-seating |
| `GRIP_SLACK_M` | `simulator/core/controller.py` | 0.12 | grip range past an object's own half-extent |
| `EE_JUMP_M`, `OBJ_JUMP_M` | `simulator/core/controller.py` | 0.05 | seam-bridge trigger |
| floor-assist cap, pin cap | `controller.py`, `grounding.py` | 0.05 | bounded silent corrections |

**G1 (~1.3 m) and H1 (~1.8 m) need no changes** — same order of magnitude, so the
thresholds stay in a sensible band. For a robot far off that scale (a small quadruped
at ~0.3 m), scale them by `limb_reach / 0.8` or the checks become meaningless: a 0.25 m
"nothing beneath this contact" threshold exceeds the entire leg.

### Environment toggles for A/B

All default on, each disable-able: `REACH_PROJ=0`, `SEAM_BRIDGE=0`, `GROUNDING_CLAMP=0`,
`RATE_LIMIT=0`, `CONTINUITY_REF=0`, `EE_ORIENT=0`. Opt-in: `NO_DRAG=1`.
Tunables: `IK_HOME_PULL` (0.05), `IK_ORI_WEIGHT` (0.4), `IK_BACKEND` (`dls`|`mink`).

---

## 10. Scenes and environments

Scenes are **robot-agnostic** — floor and objects only, no robot. `scene_compose.py`
composes a robot with a scene at load, so any scene works with any robot.

To add one: write `assets/scenes/<env>.xml` (a static geom for fixed furniture, a
**mocap** body for something carried), then register it under `environments:` in
`robot.yaml`. Its exact geometry is parsed and injected into the prompts automatically
by `agent/utils/scene_geometry.py` — never hand-describe coordinates, they will drift.

Match `grounding` to how the skill authors its base:

| scene | grounding | because |
|---|---|---|
| flat floor | `contact` | let the framework re-seat the base from the contacts |
| raised surfaces (stairs, ledge, chair) | `"off"` | contacts sit at several heights the flat re-seat can't express |

---

## 11. Troubleshooting

| symptom | cause |
|---|---|
| Robot sinks through the floor when crouching | a home-keyframe joint sits at its limit — see [§2](#-the-home-keyframe-must-not-put-a-joint-at-its-limit) |
| Contacts float or penetrate by a constant amount | `ground_offset` wrong for that member |
| `AssertionError` on joint order at startup | `joints:` not in model `qpos[7:]` order |
| Skill runs in the wrong scene | `grounding: off` unquoted — YAML parsed it as boolean `False` |
| Limbs won't reach far | `null_space_gain` too high |
| Planner dies with `Expecting value: line 1 column 1` | the plan exceeded `llm.max_tokens` and was truncated before its closing fence — raise it |
| `⚠️ Site not found` at startup | expected for `contact_bodies` without a `site:` |

---

## See also

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — how the simulator, checks and repair loop work
- [`SKILL_GUIDE.md`](SKILL_GUIDE.md) — writing skill prompts that produce good motion
