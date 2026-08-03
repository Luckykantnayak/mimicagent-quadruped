# MimicAgent — Architecture & Pipeline Guide

MimicAgent turns a **natural-language skill description** into a **verified motion
trajectory** for a robot. An LLM agent plans the motion and writes the code that
generates it, a MuJoCo simulator executes it kinematically and records a
trajectory, a quality checker scores it against universal physical laws *and*
against invariants the agent itself declared, and the agent repairs its own code
until both are satisfied.

Two properties define the system:

- It is **morphology-agnostic**. A single per-robot spec file drives both the
  simulator and the LLM prompts. There is no `if humanoid:` branch anywhere.
- It is **decision-agnostic**. The prompts and tests encode *general laws and
  mechanisms*; every skill-specific decision — which limb uses which control
  channel, which environment to run in, what would make this motion recognizably
  correct — is made by the agent, per skill, and is never baked into a heuristic.

> This document is the long-form guide. `CLAUDE.md` at the repo root is the
> working reference for anyone (human or agent) editing the code, and is the
> authority on any point where the two disagree.

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
        │  PLANS + WRITES a skill  │◀───────│  skill, RECORDS a traj     │
        └──────────────────────────┘ traj   └────────────────────────────┘
                        │                              │
                        └──────── skill_output/<robot>/<skill>/ ─────────┘
                            (generated code + trajectory + reports)
```

- **`robots/<name>/`** — per-robot *code and spec*: `robot.yaml`, a reference
  skill `example.py`, curated `exemplars/`, and two generated artifacts
  (`summary.md`, `dynamics.json`). No meshes.
- **`assets/`** — `assets/robots/<name>/` holds the MuJoCo model + meshes (and
  owns its own `<keyframe home>`); `assets/scenes/` holds **robot-agnostic**
  environments (floor + objects, no robot).
- **`simulator/`** — the kinematic MuJoCo simulator. Given a robot and a skill,
  it plays the motion and records a trajectory.
- **`agent/`** — the LangGraph pipeline of LLM nodes that authors skills and
  drives the simulator in a self-correcting loop.

Generated artifacts land in **`skill_output/<robot>/<skill>/`** (git-ignored).

### Repository layout

```
robots/
  humanoid/
    robot.yaml        # spec (§3)
    example.py        # reference skill (the default demo)
    exemplars/        # curated hand-authored skills: SKILL_CLIMB, SKILL_CARRYBOX
    summary.md        # generated: limb lengths, reach, joint + dynamics ceilings
    dynamics.json     # generated: per-joint velocity/accel ceilings, jump height
assets/
  robots/humanoid/    # humanoid.xml + meshes + <keyframe home>
  scenes/             # flat.xml  slab.xml  stairs.xml  chair.xml  carry.xml
simulator/
  run.py              # entry point (run from repo root)
  config.yaml         # run defaults
  core/               # simulator.py  controller.py  grounding.py  recorder.py  config.py
  motion/             # base.py (BaseMotionGenerator) + loader __init__.py
  utils/              # kinematics.py  reach.py  robot_config.py  robot_dynamics.py
                      # scene_compose.py  math_utils.py  file_utils.py
  tools/              # gen_robot_spec  gen_robot_summary  reach_lint  record_video  pose_editor
agent/
  run.py              # pipeline entry (one skill per invocation)
  app.py              # Streamlit demo
  agents/code_gen/    # the graph nodes
  agents/skill_gen/   # skill formatting + the separate library-inventing pipeline
  prompts/templates/  # one shared, spec-filled template set
  utils/              # test_motion_quality  trajectory_readout  scene_geometry  robot_spec
skill_output/<robot>/<skill>/    # generated (git-ignored)
```

### Robot + scene composition

Scenes contain no robot. At load time
`simulator/utils/scene_compose.py:compose_model(robot_mjcf, scene_env)` writes a
small wrapper XML *inside* `assets/scenes/` that `<include>`s both the robot
model and the environment. Writing it there is what makes MuJoCo resolve
`meshdir` and the env include correctly. Consequence: **any scene works with any
robot**, and adding an environment never touches a robot, or vice versa.

---

## 2. Why it is morphology-agnostic

Everything a robot-specific decision could depend on is declared **once**, in
`robots/<name>/robot.yaml`, and consumed by both halves:

| What varies per robot | Where it's declared | Who uses it |
|---|---|---|
| End-effector names & bodies | `end_effectors` | skill code, prompts, IK, quality checks |
| Extra weight-bearing bodies | `contact_bodies` | IK roster, grounding, contact checks |
| Joints, order, limits | `joints` | recorder order, IK clamp, joint-limit check, prompt |
| Which joints are passive | `joints[].passive` | quality checks and IK skip them |
| Contact geometry | `*.ground_offset` | grounding, penetration / grounded checks, prompts |
| Base-height envelope | `base_height_range` | the declarable `base_height_band` invariant |
| IK rest-pose pull | `null_space_gain` | IK null-space term |
| Camera framing | `camera` | renderer |
| The MJCF text | `mjcf` | injected into prompts as `{urdf}`; source of the physical summary |
| Free-text morphology notes | `robot_card` | injected into every prompt |
| Reference implementation | `example_skill` | injected into the coder prompt |
| Scenes this robot can use | `environments` | planner's choice; executer's `--scene`/`--grounding` |
| Curated reference skills | `exemplars` | prompt catalog; full source shown to the coder |

The agent adapts because the prompt templates are filled from the spec (§6); the
simulator adapts because the controller, IK, and grounding read the spec
directly.

This repo currently ships **one** robot, the 28-DoF humanoid. The abstraction was
built and validated against quadrupeds (a legged Go2 and a wheeled Go2-W) that
have since been removed from the tree — which is why you will still find general
machinery for cases the humanoid does not exercise, all of it spec-driven rather
than robot-named:

- `joints[].passive` — unactuated joints (a wheel), excluded from IK and checks.
- `end_effectors[].wheel: true` — a planted contact that may **roll**: it is
  z-seated and bears weight, but is never XY-pinned by the grounding layer and is
  exempt from the no-slide check. Runtime and check agree by construction.
- `end_effectors[].contact: false` — an EE that is IK-driven but is not a ground
  contact (the humanoid's hands), excluded from ground/penetration checks.
- `kind: quadruped` — forces `grounding: contact` (a flat-terrain, no-float
  policy). The humanoid is `kind: humanoid` and is not affected.

---

## 3. The robot spec (`robots/<name>/robot.yaml`)

Loaded by `simulator/utils/robot_config.py:load_robot_spec` into a `RobotSpec`
dataclass. Abridged from the shipped humanoid:

```yaml
kind: humanoid
name: humanoid
scene_xml: assets/scenes/flat.xml            # default environment (bare floor)
mjcf:      assets/robots/humanoid/humanoid.xml   # injected as {urdf}
base_body: pelvis                            # free-joint body (camera + root)
sim_freq:  200
null_space_gain: 0.02        # 28 DoF: relax the rest-pose pull so limbs reach far

# Ordered end-effectors. `name` is what skill code & prompts use — no
# normalization layer. ground_offset = body-origin -> contact-point distance.
# contact: false  -> IK-driven but not a ground contact (excluded from those checks).
end_effectors:
  - {name: right_foot, body: right_foot, site: ee_right_foot, ground_offset: 0.05}
  - {name: left_foot,  body: left_foot,  site: ee_left_foot,  ground_offset: 0.05}
  - {name: right_hand, body: right_hand, site: ee_right_hand, ground_offset: 0.04, contact: false}
  - {name: left_hand,  body: left_hand,  site: ee_left_hand,  ground_offset: 0.04, contact: false}

# MAXIMAL contact roster — every body part that may bear weight. Together with
# end_effectors these form the UNION IK ROSTER (§5): give one a world target and
# IK PLACES it like a foot; return None and it is FK-only.
contact_bodies:
  - {name: pelvis,     body: pelvis,          ground_offset: 0.09}   # sit
  - {name: right_knee, body: right_shin,      ground_offset: 0.06}   # kneel / crawl
  - {name: right_forearm, body: right_lower_arm, ground_offset: 0.045}  # plank
  # ... torso, head, left_knee, thighs, upper arms ...

# Joints in MODEL qpos[7:] ORDER (the recorder and checks depend on this).
joints:
  - {name: abdomen_x, limits: [-1.0472, 1.0472]}
  # ... 28 total ...

base_height_range: [0.30, 2.20]   # wide: a flip apex and a deep landing crouch
camera: {lookat: [0.0, 0.0, 0.9], distance: 4.0, azimuth: 90, elevation: -15}

# Named environments a skill may run in. `flat` is always implied. The planner
# picks ONE by name; the executer runs the sim with that scene + grounding.
# NOTE: quote "off" — unquoted, YAML parses it as boolean False.
environments:
  - name: ledge
    scene: assets/scenes/slab.xml
    grounding: "off"
    objects: []
    description: "A raised platform ahead ..."

# Curated hand-authored reference skills, in robots/<name>/exemplars/.
exemplars:
  - name: SKILL_CLIMB
    environment: ledge
    tags: [mantle, surface-ascent, hands-on-lip, feet-smear, absolute-base]
    summary: "Mantle onto a tall ledge: hands reach over the lip ..."

example_skill: robots/humanoid/example.py

robot_card: |
  Free-text morphology description injected into every prompt.
  KINEMATIC FACTS ONLY: DoF, end-effectors, standing height, and each joint's
  axis + sign + range. It deliberately says NOTHING about which control channel
  a limb should use — that is the agent's decision, per limb per phase.
```

`RobotSpec` exposes derived views used across the code: `ee_names`, `ee_bodies`,
`ee_sites`, `ee_ground_offsets`, `contact_ee_indices`, `wheel_ee_indices`,
`contact_body_names/bodies/offsets`, the union `ik_names/ik_bodies/ik_sites`,
`joint_names` (full, model order), `actuated_joint_names`,
`passive_joint_indices`, `joint_limits`, plus the dynamics-tier
`joint_speed_ceilings`, `joint_accel_ceilings`, `jump_height`, and the resolvers
`all_environments` / `get_environment` / `get_exemplar`.

**Invariant:** the `joints` list must match the model's `qpos[7:]` order. The
controller asserts this at startup, because the recorder stores the joint vector
in that order and the quality checks index into it.

**Two generated companions** (§9, step 3), both produced by
`simulator/tools/gen_robot_summary.py`:

- `summary.md` — human-readable physical summary (limb lengths, per-EE reach,
  per-joint position limits, and — when the MJCF carries mass/inertia and torque
  limits — velocity/acceleration ceilings plus cited forward-speed / jump /
  body-rate estimates). Injected into prompts as `{robot_summary}`; keeps
  commanded amplitudes *and velocities* physically feasible.
- `dynamics.json` — the machine-readable form (`vel`, `accel`, `jump`), read by
  the `joint_velocity_envelope` check.

The dynamics tier is **automatic**: emitted when the model supports it, silently
skipped when it does not (both the summary section and the velocity check
self-omit; the robot still runs).

---

## 4. The simulator — how a skill is executed

Entry: `python simulator/run.py --robot <name> --mode <skill> --record True`
(run from the repo root). Flow through `simulator/core/`:

```
run.py
  └─ load_robot_spec(robot)                     # robots/<robot>/robot.yaml
  └─ resolve the scene:  --scene  >  the exemplar's own environment  >  spec default
  └─ MujocoSimulator(scene, robot_mjcf=spec.mjcf)   # composes robot+scene, applies "home"
  └─ MotionControllerRunner(sim, args):
        1. build IK over the UNION roster (spec.ik_sites) + actuated joints
        2. read the home base pose from qpos
        3. FK -> initial body-frame positions of the whole roster (init_ee_pos)
        4. resolve the skill:
             mode == "example"      -> robots/<robot>/example.py
             a generated skill      -> skill_output/<robot>/<mode>/  (highest version)
             a curated exemplar     -> robots/<robot>/exemplars/<mode>.py
             else                   -> the example
        5. instantiate Motion(init_ee_pos=...) and reset(base_pose)
  └─ main loop, each step:
        state = runner.step():
          motion_state = skill.step(dt)      # base pose + per-EE targets + declarations
          contact grounding                  # register base, pin contacts (§4b)
          safety nets                        # clamp / floor assist / reach seat / reach proj
          IK solve  (+ seam bridge)          # joint angles, clamped to limits
          apply skill.joint_overrides
          place mocap objects                # authored poses, then attachments() override
          write root pose + joints to qpos, FK
          compute the collision layer        # penetration, self-collision, contacts, clearance
        recorder.record(state)               # inside the capture window
  └─ recorder.save() -> two pkls
```

`--headless True` skips glfw/GL and the real-time throttle entirely — no window,
no display, ~10× faster. The agent always runs headless; only the recorded pkl
matters to it.

### 4a. Initial conditions and the capture window

The robot starts at the **`home` keyframe of its own model** (standing base
height, nominal joint angles) — the model owns it, not the scene. The controller
computes each roster member's body-frame position via forward kinematics and
hands it to the skill as `init_ee_pos`.

The capture window in `run.py` is fussier than it looks, and every part earns its
place:

1. **50 warm-up frames** — IK settles from the raw keyframe.
2. The skill's phase clock is reset to 0 and the seam bridge cleared, so the
   recording starts at a true phase 0 with no loop seam. Essential for one-shot
   skills: a pick-and-place would otherwise teleport its object at the wrap.
3. **A 0.3 s settle window** pins the phase at 0 while IK converges to the exact
   phase-0 pose. Without it the *achieved* contacts drag toward their pins over
   the first ~0.1 s and that reads downstream as a phantom `contact_no_slide`
   drift.
4. **`record_cycles / skill_freq × sim_freq` frames.** The rate is the *skill's*
   own `freq`, not the config's, so a slow skill is captured in full rather than
   truncated. `--record_cycles` comes from the plan's `repetitions`: a cyclic
   skill authors ONE cycle and the framework records N of them, so a walk travels
   several strides instead of one stride stretched over the whole clip.

The settle bound is inclusive (`<= start_frame`) because `base.step()` reads `t`
*then* increments it. Off by one and the last recorded frame lands on phase 1.0,
which wraps to 0 — snapping a one-shot skill's final frame back to its start pose
and manufacturing a fake "0 cm net motion", a phantom stance-foot slide, and a
seam teleport that then poisons the repair loop.

### 4b. Contact grounding (`simulator/core/grounding.py`)

The framework computes the parts of the motion that *geometry* determines, rather
than asking the LLM to get them right by hand. No gravity, no dynamics, no
control — pure geometry. Two modes, chosen per environment:

- **`--grounding contact`** — the framework registers the base along gravity so
  the lowest active contact rests on the floor, and XY-**pins** each planted
  contact so it cannot slide. The skill authors the gross base pose; the
  framework derives only the along-gravity registration.
- **`--grounding off`** — the layer is bypassed entirely; the skill owns the base
  absolutely. Required whenever contacts live at more than one height (feet on
  the floor *and* hands on a ledge), which flat-floor registration cannot express.

The contact set is **per-phase**, from the skill's `contacts(phase)`. That single
mechanism covers a fixed support set (walk), a changing one (a cartwheel: feet →
hands → feet), and a flight window (`[]` → ballistic).

One asymmetry matters: when the skill **explicitly declares** contacts, they are
load-bearing *by declaration* and get seated even if authored a few cm high; only
the `active is None` fallback uses the "all feet well above the floor ⇒ probably
airborne" heuristic. That is what stops a crawl whose hand/knee targets sit
slightly high from being misread as a hover. Registration snaps **up**
immediately when a contact is at or below the floor (so a fast foot-plant is
never left penetrating) but eases **down** rate-limited when a higher contact
takes over (so a hand-off cannot pop the body).

### 4c. Inverse kinematics and the safety nets

`simulator/utils/kinematics.py` is generic damped-least-squares over the spec's
sites and actuated joints. It is N-target generic and **clamps every joint to its
model range each iteration**, so recorded trajectories respect joint limits by
construction. (`IK_BACKEND=mink` selects an opt-in QP solver instead; DLS is the
default — see `make_ik` for the measured trade-off.)

Three properties of the solve matter more than the solver choice, and each was
added because a specific failure mode traced back to its absence:

- **A physical slew limit.** No actuated joint may change faster than its
  torque-derived ceiling (`dynamics.json` × dt). It is applied to the *final* joint
  vector in `controller.step()` — after IK, after `joint_overrides`, after the seam
  bridge — since that is the only point that sees every path capable of stepping a
  joint. This makes `no_teleport` structurally impossible rather than checked after
  the fact, and it was by far the largest consumer of the repair loop. It is
  surgically selective (~0.07% of steps on a walk) and its magnitude is recorded as
  `rate_limit_mag`, so a skill demanding impossible speed under-reaches *visibly*
  instead of snapping.
- **A continuity-biased null-space reference.** Position-only IK gives 3 constraints
  against ~7 DoF per limb, so the twist axes are nearly unobservable in the residual
  and follow the null-space reference almost entirely. Pulling them toward the
  *standing* pose is right for a walk and actively harmful for an inverted one, where
  home is maximally wrong. The reference is therefore the previous frame's solution
  (which the solver already warm-starts from), with only a weak pull home so a
  redundant DoF cannot drift, and the skill's declared posture overriding both.
- **An optional rotational task.** An end-effector with an orientation target
  contributes 6 Jacobian rows instead of 3, which is what finally *observes* those
  twist DoFs. A declared contact defaults to being laid **flat on the surface beneath
  it** — the alignment error is `a_current × a_target`, which vanishes for any twist
  once aligned, so the foot's yaw stays free where a full quaternion would wrongly pin
  it. Free limbs get no orientation: forcing it on a gesture would dissolve the shape
  the joint channel exists to preserve. Skills opt into full 3-DoF orientation via
  `compute_ee_orientation_world_frame`.

Around it sit several general safety nets. All of them are *silent corrections*,
so their magnitudes are recorded and surfaced to the repair loop (§7) — a
framework that quietly fixes impossible authoring would hide the bug from the
agent that must learn to stop writing it:

| Net | What it does | Cap |
|---|---|---|
| **IK target clamp** | No EE's contact point may be aimed below its surface — a swing limb cannot be driven through the floor | — |
| **Floor assist** | Under `grounding: off`, lift the base so the deepest declared contact rests on its surface | 5 cm, lift-only |
| **Reach seat** | Seat the base along z so a declared contact stays within limb reach | `REACH_SEAT_CAP` = 0.15 m; beyond that the authored base is grossly wrong and the contact is left to float **visibly** |
| **Reach projection** | A *free* (non-contact) EE whose world target is beyond reach is pulled onto the reachable ball — otherwise IK flips configurations frame to frame (the "hand flicker") | — |
| **Seam bridge** | Crossfades a discontinuity at a control-channel handoff | re-arms on the authored **target** jump, not tracking error, so a chronically-unreachable target no longer sawtooths |

`REACH_PROJ=0`, `SEAM_BRIDGE=0`, `SEAM_BRIDGE_JOINTS=0` disable individual nets
for A/B debugging. All default on.

The reach model itself lives in `simulator/utils/reach.py` and is shared by the
runtime nets, the pre-sim lint (§6), and the physical summary — so the lint's
verdict and the runtime's behaviour cannot disagree.

### 4d. Outputs (`simulator/core/recorder.py`)

Two files in `skill_output/<robot>/<skill>/`:

- `<skill>.pkl` — evaluation trajectory: `root_pos (T,3)`, `root_quat (T,4)`,
  `joints (T,J)`, `ee_positions (T,N,3)` (+ `_achieved`), plus `fps`, `robot`,
  `ee_names`, `support_ees`, **the collision layer** (`surface_penetration`,
  `self_collision`, `contact_mask`, `surface_clearance`, `object_pos`) and the
  **intervention magnitudes** (`clamp_mag`, `reach_seat_delta`,
  `floor_assist_lift`, `reach_proj`). Consumed by the agent's quality checker.
- `<skill>_mimickit.pkl` — MimicKit motion `{fps, loop_mode, frames}` where
  `frames = [root_pos(3), exp_map(3), joints(J)]`, for downstream imitation
  learning.

### 4e. Debug tools (`simulator/tools/`)

`reach_lint.py` and `record_video.py` both instantiate the **real**
`MotionControllerRunner` — which is the point: a debug tool that modelled the sim
separately would drift from it.

```bash
python simulator/tools/reach_lint.py    --mode SKILL_X --robot humanoid --grounding off
python simulator/tools/record_video.py  --robot humanoid --mode SKILL_X --out /tmp/x.gif
python simulator/tools/pose_editor.py   --robot humanoid   # interactive stance authoring
```

---

## 5. The skill contract (`simulator/motion/base.py`)

Every skill subclasses `BaseMotionGenerator`. `base.py`'s source is injected
verbatim into the coder prompt, so the contract the LLM sees always matches the
real one — **keep its docstrings accurate**.

### Three control channels, chosen per limb per phase

The single most important idea in the contract: **nothing prescribes a channel
for any limb.** The discriminator is *coupling*, and the agent decides.

| Channel | Method | Use when |
|---|---|---|
| **World-frame IK** | `compute_ee_position_world_frame(ee, phase)` | the WORLD fixes the endpoint — a contact on a surface, or a limb that must coincide with an object it grips |
| **Body-frame IK** | `compute_ee_position_body_frame(ee, phase)` | the target is relative to the torso — a swing foot |
| **Joint** | `compute_joint_overrides(phase)` | the limb's *configuration* is the point and its endpoint is unconstrained — a gesture, a kick |

Both IK methods default to `None`, and a `None` target makes IK **skip** that
limb — which is what hands it to the joint channel. The rule of thumb the prompts
teach: a limb whose endpoint is fixed by the world or by a thing it holds → world
IK; a limb whose *shape* is the point → joints. Driving a gesture through IK
dissolves it into an arbitrary reach that hits the point but does not look like
the gesture.

### The union IK roster

`init_ee_pos` and all three channels span **end-effectors + declared contact
bodies**. So a mid-limb contact (a knee, a forearm) is *placed* by returning its
world contact point and IK bends the limb to it — it is not FK-only. A crawl
plants its knees, a plank its forearms, a sit rests its pelvis.

The reach-seat and reach-projection nets stay EE-only, since their base→tip reach
model is meaningless for a mid-limb body; a misplaced contact body simply floats
and `contact_grounded` catches it.

### Declarations

Beyond geometry, a skill *declares* things the framework and the checks then rely
on:

```python
def contacts(self, phase):        # -> ["left_foot", ...]  load-bearing this phase
    ...                           #    [] means ballistic flight
def compute_object_poses(self, phase):   # -> {name: (world_pos, quat)} for mocap objects
    ...
def attachments(self, phase):     # -> {object: [ee, ...]} rigidly HELD this phase
    ...
```

`attachments()` is the one worth calling out. The sim is kinematic — there is no
grasp physics — so a carried object has to be *placed*. Rather than have the
skill guess where an IK-solved or joint-driven hand ended up, `attachments()`
places the object at its holders' **achieved (post-IK) positions**. The grip is
then correct **by construction**: `{"box": ["left_hand", "right_hand"]}` puts the
box at the hands' midpoint, wherever those hands actually went.

### Helpers

- `reach_limited_target(ee, desired_world_pos)` — returns a target the limb can
  actually reach (the desired point, or the closest point on the reachable ball).
  Use it to *construct* world targets far from the standing pose, so the contact
  is reachable by construction rather than silently under-reached.
- `limb_root_world(ee)` — world position of the limb's root (a hand's shoulder, a
  foot's hip).
- `advance_base_planar(forward, lateral, dt)` — translate the base along the
  **heading (yaw-only)** frame. Do *not* rotate a forward velocity by the full
  `root_quat`: it includes pitch, so a near-horizontal crawl (pitch ~70°) leaks
  most of its speed into the discarded z-axis and barely advances.

### Frame conventions

- End-effector targets may be authored in **either** the body frame or the world
  frame, per limb per phase.
- The base pose is **always** world frame.
- `phase ∈ [0,1]` drives everything, and may be **cyclic or one-shot**.
- `self.freq` is the phase *rate*: one pass through phase 0→1 takes `1/freq`
  seconds. It sets the skill's wall-clock duration and therefore every joint and
  base velocity, so it is a free per-skill choice picked from the motion's real
  tempo. The default of 1.0 plays the entire skill in one second, which is far too
  fast for anything with several events.

> **Historical note.** Earlier versions of this contract mandated body-frame EE
> targets and a base driven *only* by velocity integration. Both restrictions are
> gone: absolute world-frame base trajectories and one-shot (non-cyclic) skills
> are first-class, which is what makes stair-climbing, mantling, and
> pick-and-place expressible at all.

### Curated vs generated

A small number of hand-authored reference skills are **tracked** as
`robots/<name>/exemplars/<NAME>.py` and registered in `robot.yaml`. Agent output
lives in git-ignored `skill_output/<robot>/` and is regenerable.
`robots/<name>/example.py` is the default demo. The humanoid ships two exemplars
— `SKILL_CLIMB` and `SKILL_CARRYBOX` — deliberately minimal; this is not a skill
library.

---

## 6. The agent pipeline (`agent/`)

A LangGraph state machine (`agent/run.py:build_graph`) over
`agents/code_gen/state.py:AgentState`:

```
skill_filter → motion_planner → code_generator → code_executer → unit_test_router
                    ↑                                 ↑                 │
                    │                            critic_coder ← motion_diagnosis
                    │                                 ↑                 ↑
                    └── replan ──────────────────  (repair)      critic_router ← unit_test
                                                                        │
                                                       finalize (best version) → END

unit_test_router:  clean run          → unit_test
                   lint_fail_streak≥2 → motion_planner   (REPLAN, once)
                   crash / lint fail  → critic_coder     (repair)
                   out of iters       → finalize
critic_router:     rewrite_mode=none  → finalize
                   stuck              → motion_planner   (REPLAN, once)
                   otherwise          → motion_diagnosis → critic_coder
```

`unit_test_router` and `critic_router` are pure routers (`lambda state: state`
plus a conditional edge). The loop is bounded by `critic.max_iters` (7 — the
physics floor alone can consume 5, and the one-shot replan needs room after it).

**Node by node:**

- **skill_filter** — an LLM feasibility gate, run before any compute is spent.
  `SKIP_SKILL_FILTER=1` bypasses it for explicit user-requested `--skill` runs
  (aerial or expressive moves it would otherwise veto).
- **motion_planner** — emits one JSON plan: the `environment` to run in,
  `cyclic`, `duration_s` + `repetitions`, `phase_definition.sub_phases` (the
  canonical timeline), a `contact_schedule` (per-phase declared contacts,
  surface, `planted|pivot` mode), `free_limb_mode` per free limb
  (`world_reach | body_swing | joint_posture | none`), `fidelity_invariants`, and
  `sliding_contacts`. `contact_schedule`, `free_limb_mode`, and
  `base_motion.per_phase` must all tile [0,1] with exactly the sub-phase
  boundaries.
- **code_generator** — writes the `MotionGenerator` subclass. Before the call,
  `coder_agent.py` deterministically expands the plan: `_channel_table` turns the
  plan's channel choices into an explicit per-limb method table, `_environment_context`
  adds the scene's objects + grounding + an `attachments()` recommendation, and
  `_plan_lint` flags internal contradictions (an object is carried but no limb is
  world-bound to grip it) and licenses the coder to override a mislabelled
  channel. The coder is also shown the **full source** of a curated exemplar whose
  environment matches.
- **code_executer** — writes `skill_output/<robot>/<skill>/<skill>_NNNN.py` (plus
  the exact prompt that produced it), runs the **pre-sim lint** (below), then
  runs the simulator as a subprocess with the resolved scene, grounding, and
  cycle count.
- **unit_test** — scores the recorded trajectory (§7) and produces the motion
  readout.
- **motion_diagnosis → critic_coder** — explains the failure, then rewrites,
  incrementing the file version.
- **finalize** — §8.

### The pre-sim reachability lint

Between writing the file and paying for a rollout, `run_reach_lint` invokes
`simulator/tools/reach_lint.py`: it rebuilds the same controller, replays the
skill's `step()` over one cycle with **no IK and no rendering**, and asks whether
every declared world contact is within its limb's reach from the authored base.
Milliseconds, versus a minutes-long doomed rollout.

Only a **hard** violation — one the runtime reach-seat cannot absorb —
short-circuits; softer ones fall through to the sim, which handles them. A hard
fail returns a signature of *what* it caught (`arr:<ee>@<phase>`, `incompat@`,
`horiz:`, `notgt:`). Two consecutive **identical** signatures mean the coder is
patching around a structural *plan* bug, so the router replans instead of looping
the coder on a schedule the code cannot fix. A clean sim run resets the streak.

It also hard-fails a **chronically unreachable FREE limb**, gated on severity *and*
persistence (beyond 1.25× reach on more than 10% of frames). A brief over-reach is
absorbed by the runtime projection and stays a soft warning; a sustained one means
the skill's intent is silently truncated for most of the motion. This is calibrated
against real runs: `example`, `STAIRS` and `CRAWL` score 0.0, while a handstand whose
hands sit at 1.99× reach on 13.9% of frames and a cartwheel at 43–45% are both caught
in milliseconds — the two skills that previously consumed 18 and 17 iterations without
ever converging.

The arrival check runs in both grounding modes; only the reachability check
self-gates to `grounding: off`.

### Prompts are one shared, spec-filled template set

`agent/prompts/templates/*.txt` (`coder`, `motion_planner`, `critic_coder`,
`motion_failure_diagnosis`, `skill_filter`, `skill_formatter`, `skill_gen`), each
with `=== SYSTEM ===` / `=== HUMAN ===` sections. `prompts/prompt_loader.py` fills
morphology placeholders from the `RobotSpec` via `str.replace` — never `.format`,
because the templates contain literal JSON braces:

`{robot}` · `{urdf}` · `{end_effectors}` · `{num_end_effectors}` ·
`{support_set}` / `{free_ees}` · `{contact_bodies}` · `{contact_offsets}` ·
`{joint_order}` · `{joint_limits}` · `{base_height_range}` · `{morphology_card}` ·
`{example_skill}` · `{base_class_source}` (read live from
`simulator/motion/base.py`) · `{robot_summary}` · `{environments}` · `{exemplars}`

Two of these carry more weight than their size suggests:

- **`{contact_offsets}`** is an explicit per-contact `ground_offset` table *plus
  the arithmetic*: a declared contact's world z is `surface_height +
  ground_offset`, never the bare surface height. This exists because authoring a
  foot at z=0 on the floor — which drives the sole a `ground_offset` *below* it —
  was the single most-repeated authoring bug.
- **`{environments}`** annotates each scene with its **real geometry**, parsed
  straight out of the scene XML by `agent/utils/scene_geometry.py`: per-step tread
  heights, slab extents, box pose, distance from the robot's home keyframe.
  Before this the agent got only a name and a hand-written blurb, so it *guessed*
  — it climbed 2 of 5 steps, planted feet where no tread was, and modelled a
  finite staircase as a cyclic gait.

Per-call placeholders (`{{MOTION_SPEC_INPUT}}`, `{SKILL_NAME}`, `{USER_SKILL}`,
`{CURRENT_SKILL_CODE}`, …) are filled later by the nodes. Adding a robot needs
**no** prompt edits.

---

## 7. What a generated skill must satisfy

Enforced by `agent/utils/test_motion_quality.py:evaluate_trajectory`, all
parameterized from the `RobotSpec`. There are **three tiers**, and the
distinction is what stops the loop from either shipping a legal-but-wrong motion
or Goodharting its own tests.

### Tier 1 — the physics floor (always on, drives the repair loop)

| Check | Rule |
|---|---|
| `surface_penetration` | no robot geom centre below the floor or inside a static scene box |
| `self_collision` | **soft**: gross overlap only, over non-adjacent geom pairs (the model filters self-collisions, so `mj_collision` cannot see them) |
| `support_or_ballistic` | at every frame *some* declared contact rests on its surface, **or** the skill declares flight (`contacts == []`) and the base is ballistic (accel ≈ −g) |
| `contact_grounded` | **every** declared contact actually rests on its surface, not just some of them |
| `contact_no_slide` | over each firmly-planted stance run, net horizontal drift < 10 cm |
| `joint_limits` | every actuated joint within its model range |
| `no_teleport` | no actuated joint steps discontinuously between adjacent frames — now largely enforced UPSTREAM by the slew limit (§4c), so this is a backstop rather than the loop's main workload |
| `ee_no_teleport` | the EE-space companion: no base-compensated ~1 m/frame jump of a free limb |
| `joint_velocity_envelope` | no joint *sustains* a speed above its torque-derived ceiling (p90 > 1.5× `dynamics.json`) |
| `framework_interventions` | a **saturated** silent correction (§4c) means the authoring was impossible. The recorded set is `clamp_mag`, `reach_seat_delta`, `floor_assist_lift`, `reach_proj`, `pin_mag` (the bounded contact hold) and `rate_limit_mag` (slew held back by the physical ceiling) |
| `clip_duration` | a sub-2 s recording is a snippet, not usable motion data |

Several of these encode a distinction that took real failures to find:

- `support_or_ballistic` passes as long as *one* declared contact rests, so a
  hands-and-knees crawl that declares its knees and then floats them still passed
  it. `contact_grounded` is the per-contact companion that catches exactly that.
- `no_teleport` fires on an **isolated** single-frame spike above a physical slew
  ceiling (or on any spike ~4× the ceiling), via local-neighbour isolation — so
  sustained fast-but-smooth motion and the seam bridge's own crossfade are not
  flagged. It catches the control-channel-handoff pop: the abrupt pose change a
  semantically-correct skill still shows.
- `joint_velocity_envelope` catches *sustained* infeasible velocity a policy could
  not track, which is a different failure from `no_teleport`'s lone spike. A brief
  whip or kick passes.
- The ceilings are per-joint and **fps-aware** (rad/s × dt), not an fps-blind
  rad/frame floor.

`base_height_envelope` was **demoted** out of this tier into a declarable
invariant: a crouch, crawl, or jump legitimately leaves any fixed band, and its
one real job (hover or burial) is already covered by support + penetration.

### Tier 2 — gated tests

Each self-skips when out of domain, and the gate is **only ever a physical
declaration** — a `[]`-contact segment, scene objects present — never an
editorial skill label:

`flight_window` (judged as the ballistic rise above the *takeoff surface*, so a
hop onto a slab is terrain-independent) · `object_grip` (a lifted object must stay
coincident with *some* end-effector — morphology-agnostic, no name matching) ·
`object_continuity` · `object_at_rest` · `object_penetration`.

There are **no skill-label test gates**. The old `motion_type` → `foot_clearance`
gate and eight superseded evaluators were deleted: prompts and tests are general
laws and mechanisms, and the agent makes every skill-specific decision. ("A
support foot must lift" is a stepping-gait assumption, not a general law — a
static balance legitimately keeps its feet planted, and a dragging foot is
already caught by `contact_no_slide`.)

### Tier 3 — agent-declared fidelity invariants

The "does it actually *look* like the intended motion?" layer. The planner
composes a handful per skill from eight general geometric primitives:

`net_progress` · `peak_rise` · `swing_lift` · `phase_offset` ·
`base_height_band` · `object_coincident` · `end_state` · `stance_no_drift`

The primitives are universal relations; all the skill-specificity is in *which*
ones the planner picks and with what parameters — exactly like the control-channel
decision.

**Governance matters here.** Invariants are **planner-owned** and are read from
`motion_plan_*.json`, never from the code — otherwise the critic-coder would be
the repairer editing the test it keeps failing. If diagnosis concludes an
invariant is itself wrong, the loop must go through the **replan** edge and have
the planner re-declare it. The planner likewise owns `sliding_contacts` (a full
exemption from the no-slide law) and per-contact `mode: pivot` (a *bounded* drift
budget, so a real pirouette passes but a runaway "pivot" still fails).

### Perception before verdict

`run_unit_test` also builds a **motion readout**
(`agent/utils/trajectory_readout.py`): a few paragraphs of pure geometry — base
path, per-limb contact timeline with lift and drift, gait phase offsets, object
path — fed to diagnosis and the critic. It describes; it does not judge.

This exists because the loop's deepest flaw was that the critic never saw the
motion, only a vector of pass/fail lamps, so it local-patched and the checks
fought each other. A per-version history line (score, floor failures, fidelity
met) makes cross-version oscillation visible too — a five-iteration thrash that
was invisible frame-by-frame is obvious across versions.

**A skill is a data object** — the JSON the pipeline consumes, produced from free
text by `skill_formatter`:

```json
{
  "skill_id": "SKILL_WALK",
  "skill_name": "Steady Walk",
  "skill_intent": "Walk forward at a steady pace",
  "skill_description": "Phase 0-0.5: left foot swings ... Phase 0.5-1: ...",
  "think": "reasoning (stripped before code-gen)"
}
```

---

## 8. The repair loop: scoring, stalling, and shipping

**The loop continues while EITHER tier is unmet.** `rewrite_mode` is `structural`
(floor broken) → `fidelity` (floor clean, but it does not look like the skill) →
`none`. Without the fidelity tier *driving* repair, the loop exited the instant
the floor was clean: a carry-box run that had picked the box up but let it drift
out of the hands shipped at 4/6.

**Version selection.** Each version is scored:

```
generic_pass·1000  +  passed_all_discrete·100  +  40·fidelity_fraction
                   +  reward  −  core_severity
```

The tiers strictly dominate, so an invariant can never outrank a floor failure
(40 < 100 < 1000). `core_severity` (< 1) is the normalized magnitude of the
*failed* core checks: it breaks reward ties so a nearly-fixed version outranks a
badly-broken one with the same failure **count** — without it the loop once
shipped v1-worst over v6-repaired. A `>=` tie-break keeps the latest.

There is deliberately **no early stop-loss**: a fidelity fix that briefly
regresses something must still be allowed to keep experimenting. Termination is
the iteration cap, the one-shot replan, and finalize.

**Stall detection.** One flag, `stuck`, set in `run_unit_test` and read by *both*
`critic_router` (the replan decision) and the planner (the replan context), so the
two can never disagree. It fires when the overall score has plateaued for ≥2
iterations **or** any single core test has been failing ≥3 versions in a row.

The per-test streak exists because the overall score can tick up on an unrelated
severity change while one specific check stays dead for iterations — a carry-box's
`object_grip`, a crawl's `contact_grounded` six times running — so the plateau
counter kept resetting and the loop never replanned.

A replan is fed the previous plan, the readout, and the version history, so it
*diffs* rather than starting blind. It is capped at once per skill. This is the
"step back and re-decide" that a code rewrite structurally cannot do: a wrong
control channel, or an invariant that was itself wrong.

**Finalize** is where a run becomes a deliverable, in four steps:

1. Copy the **best-scoring** version — not the last — to a new highest number,
   since the loader always takes the highest.
2. **Always** re-run the sim under the resolved environment, and warn loudly if
   that re-run fails. This is what stops a stale pkl from a different version
   shipping silently beside the wrong code.
3. Render a preview GIF (best-effort; never fails the run).
4. Ship clean `<skill>.py` + `<skill>_prompt.md` + `<skill>_plan.json` and
   **delete every per-iteration intermediate**.

So a finished directory looks nothing like a mid-run one. The loader handles
both: `simulator/motion/__init__.py` prefers a versioned `<skill>_<N>.py` when
present (a run in progress) and falls back to the clean `<skill>.py` (finalized).

---

## 9. Adding a new robot or morphology

No code or prompt changes — just a new model, a spec, and a reference skill.

> **The step-by-step guide lives in [`ADDING_A_ROBOT.md`](ADDING_A_ROBOT.md)** — MJCF
> requirements, every `robot.yaml` field with its format and failure modes, the
> `example.py` contract, the scale-dependent constants, a verification checklist and
> a troubleshooting table. This section is the summary; that file is the reference.

1. **Drop the MuJoCo model + meshes under `assets/robots/<name>/`.** Follow the
   MJCF format standard, which is what makes the spec auto-scaffoldable and the
   physics extractable:
   - the root body carries the `<freejoint>`;
   - each end-effector body has a `<site name="ee_<name>"/>` — this marks it as an
     EE for auto-detection *and* gives IK a real target (no site ⇒ IK falls back
     to the body centre);
   - each actuated joint has a `range` (position limits) and `actuatorfrcrange`
     (torque, N·m); passive joints have no actuator;
   - a `<keyframe name="home">` holds the rest pose;
   - mass/inertia come from `<inertial>` tags or geom density.

2. **Auto-scaffold the spec:**
   ```bash
   python simulator/tools/gen_robot_spec.py --robot <name>
   ```
   Exact from the MJCF: `joints` (qpos order + limits), `base_body`, passive
   flags, `robot_card`, paths. Marked `# REVIEW` for you to check:
   `end_effectors` (from `ee_*` sites, else a leaf-body heuristic — fix the order,
   `ground_offset`, and `contact` flags), `base_height_range`, `camera`. Then add
   `contact_bodies` for any part that may bear weight, and fill `environments:` /
   `exemplars:` as you author scenes and reference skills (neither is derivable).

3. **Generate the physical summary and ceilings:**
   ```bash
   python simulator/tools/gen_robot_summary.py --robot <name>
   ```
   Writes `summary.md` (injected as `{robot_summary}`) and `dynamics.json` (read
   by `joint_velocity_envelope`). The dynamics tier is automatic — emitted when
   the MJCF carries mass/inertia and a torque source, skipped otherwise. Override
   with `--no-dynamics` / `--dynamics`.

4. **Write `robots/<name>/example.py`** — a hand-authored `BaseMotionGenerator`
   subclass. It is the default demo *and* the style reference injected into the
   coder prompt, so make it clean and make it demonstrate the channel choices you
   want imitated.

5. **Point the agent at it**: set `robot.name: <name>` in `agent/cfg/config.yaml`
   (or export `ROBOT_NAME=<name>` to override per process, which lets two robots
   generate in parallel without editing the shared config).

6. **Verify**: `python simulator/run.py --robot <name>` plays the example
   immediately, composed with `assets/scenes/flat.xml`.

**Adding an environment** is independent of any robot: write
`assets/scenes/<env>.xml` (a mocap body for a carried item, a static geom for a
fixed one) and register it under `environments:` in `robot.yaml`. The planner
picks from registered environments; a request for an unregistered one prints a
`⚠️ MISSING ASSET` warning telling you to add it.

An environment's `grounding` **must** match how its skills author the base: `off`
for skills that set an absolute base with world-frame contacts (stairs, ledge,
chair, carry — raised or object surfaces the flat re-seat cannot express), and
`contact` for flat-floor skills that let the framework re-seat.

**Non-IK articulation** has two options, neither needing new machinery: register
the limb tip as another **end-effector** (mark it `contact: false` if it is not a
ground contact — a skill then authors its position and IK solves the chain), or
drive its joints directly via `compute_joint_overrides`. Which one is right is a
per-skill, per-phase decision (§5), not a property of the robot.

---

## 10. Worked example — carrying a box

`robots/humanoid/exemplars/SKILL_CARRYBOX.py` is the clearest demonstration of the
contract, because it needs every part of it at once.

The environment is `carry_box`: `assets/scenes/carry.xml` adds a **mocap** crate
resting on the floor ahead, and grounding is `off` — the skill must *squat* to
reach the box, which contact registration would cancel by re-seating the base up
to the feet. `freq = 0.3`, so one pass takes ~3.3 s.

Its organizing decision is that **the box is the primary trajectory**. A single
`_box_pos(phase)` function defines where the crate is at every moment — resting,
lifting, carried at chest height, lowering, placed — and everything else is
derived from it:

- **The hands are world-frame IK**, targeting `box ± grip_y` on its two side
  faces. This is the coupling principle in action: the hands' endpoints are fixed
  by a thing they must coincide with, so they are world-bound rather than
  joint-driven. Wherever the box goes the hands follow. There is no grasp physics
  — while held, the box simply *is* where the hands are. After the release phase
  the hands return `None`, which drops them out of IK and frees the arms.
- **The feet are also world-frame**, and the comments explain why the details
  matter: during the squats they plant narrow and under the pelvis so the reaching
  hands clear the thighs; during the carry each foot holds a **fixed world x**
  through its stance and only then swings ahead, so a planted foot never drags as
  the pelvis advances.
- **The base is absolute** — `update_base_motion` writes `root_pos` and
  `root_quat` directly (pelvis x, height, and forward hinge), rather than
  integrating a velocity. This is exactly the authoring style the old body-frame /
  velocity-only contract could not express.
- **`contacts(phase)` is per-phase**: both feet through the squats, then only the
  foot currently in stance during the carry, keeping ≥1 support at all times.

Note what it does *not* use: `attachments()`. The framework offers that as the
alternative topology — declare `{"box": ["left_hand", "right_hand"]}` and the box
is placed at the hands' *achieved* post-IK midpoint, making the grip correct by
construction. That is what the prompts recommend to generated skills, since it
removes all open-loop guessing. This exemplar predates it and instead drives the
box open-loop while aiming the hands at the same points, which works because both
are computed from the same `_box_pos`. Both topologies are legitimate; the
attachment one is harder to get wrong.

What the checks then verify: `object_grip` confirms the lifted crate stayed
coincident with *some* end-effector (no name matching — it is morphology-agnostic);
`object_at_rest` confirms it did not spawn mid-air; `contact_grounded` confirms
the declared foot contacts really rested through the squat; and the planner's
`object_coincident` and `end_state` invariants confirm it went where the skill
intended. `SKILL_CLIMB` is the companion example for multi-height contacts —
hands over a ledge lip, feet smearing the vertical face, base rising onto the top.

---

## 11. Quick reference

| Task | Command (from repo root) |
|---|---|
| Play a robot's reference skill | `python simulator/run.py --robot <name> --record True` |
| Play a generated skill or exemplar | `python simulator/run.py --robot <name> --mode <SKILL> --record True` |
| Play in a specific scene | `... --scene assets/scenes/stairs.xml --grounding off` |
| Generate one skill from a prompt | `cd agent && python run.py --skill "..." --name <SKILL>` |
| Interactive demo | `cd agent && streamlit run app.py` |
| Scaffold a robot spec | `python simulator/tools/gen_robot_spec.py --robot <name>` |
| Generate summary + dynamics | `python simulator/tools/gen_robot_summary.py --robot <name>` |
| Debug an out-of-reach contact | `python simulator/tools/reach_lint.py --mode <SKILL> --robot <name> --grounding off` |
| Render a video/GIF | `python simulator/tools/record_video.py --robot <name> --mode <SKILL> --out x.gif` |

Environment toggles: `SKIP_SKILL_FILTER=1` (bypass the feasibility gate) ·
`ROBOT_NAME=<name>` (per-process robot override) · `REACH_PROJ=0`,
`SEAM_BRIDGE=0`, `SEAM_BRIDGE_JOINTS=0` (disable individual safety nets for A/B
debugging).

Key files: `robots/<name>/robot.yaml` (morphology) · `simulator/motion/base.py`
(skill contract) · `simulator/core/controller.py` (execution) ·
`simulator/core/grounding.py` (contact layer) · `agent/run.py` (pipeline) ·
`agent/prompts/templates/` (prompts) · `agent/utils/test_motion_quality.py`
(the three test tiers) · `agent/utils/trajectory_readout.py` (the motion readout).

There is **no test suite**. `agent/utils/test_motion_quality.py` is not pytest —
it is the trajectory-quality checker the pipeline runs at its unit-test node.
