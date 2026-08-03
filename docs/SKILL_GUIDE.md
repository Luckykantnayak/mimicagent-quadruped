# Writing Skills That Work

How to phrase a `--skill` description so the agent produces the motion you actually
want — with emphasis on scene-interaction skills, which are the ones most sensitive to
how you ask.

```bash
cd agent && python run.py --skill "<description>" --name SKILL_NAME
```

---

## The one rule

**The framework is limited by your description, not by its own capability.**

Every physical law is enforced and every violation is reported. What no law can know is
*which* legal motion you meant. If a skill comes back doing something legal but not what
you pictured, the fix is almost always the prompt.

A measured example. Box-carry failed every attempt across a long session — the hands
stopped 33 cm above the crate and the crate levitated to meet them. Framework fixes did
not solve it. This description did, on unchanged code:

> ❌ "walks up to a crate resting on the floor, **bends down** and grips it with both hands, lifts it, carries it forward"
>
> ✅ "walks up to a crate resting on the floor, crouches down and grips it with both hands, **lifts it and stands back up to full height, carries it forward while standing upright with the crate held at waist level**, then crouches down again and sets it back on the floor"

Result: hands reached 0.199 m against crate faces at 0.12 m, a genuine grip (the
framework's grip gate never fired), and a clean physics floor.

---

## What to specify

### 1. Where the robot starts, when it matters

The robot always begins at its model's home keyframe, on the floor. If the skill needs
it somewhere else, **say how it gets there**.

> ❌ "walks to the edge of a raised ledge, steps off and drops down"
> — implies it is *already on* the ledge, so the agent authored ledge height at floor x
> and the robot walked on thin air.
>
> ✅ "walks up to a raised block, **climbs on top of it**, walks across to the far edge, then jumps off and lands"

### 2. Posture, if you care about it

Walking in a crouch is not a physics violation, so **nothing catches it**. Stair-climbing
came back hunched (torso 0.60 m above the stance foot, against 0.83 upright) and scored a
clean pass. Adding "**standing upright with its torso tall and legs extending on each
step**" produced 0.82 — at no cost to the score.

Say it if you want it: *upright*, *torso tall*, *arms extended*, *knees driving high*.

### 3. Phases, in order

The planner builds a phase timeline. Describing the motion as a sequence gives it the
structure directly: *approach → crouch → grip → lift → carry → set down*. Vague verbs
("interacts with", "handles") produce vague phase boundaries.

### 4. What the end state should be

"…then stands up", "…and lands on both feet", "…returning to the starting pose". The
planner turns these into `end_state` invariants that the repair loop then enforces.

---

## What to leave out

**Don't prescribe control channels or joint angles.** Choosing world-IK vs body-IK vs
joint overrides per limb per phase is the agent's job, made from a general coupling
principle. Prescribing it removes the reasoning that makes the system general.

**Don't supply coordinates.** Scene geometry — exact surface heights, footprints,
distances from the robot's start — is parsed from the scene XML and injected
automatically. Numbers you type may contradict it.

**Don't describe fixes to bugs you noticed.** If a skill has a flaw, that is signal
about the framework or the description's *intent*; encoding the fix produces a skill
that works for reasons you can't reuse.

---

## Scene-interaction skills

The planner picks one registered environment by name from its description. Make the
target obvious — say "crate", "chair", "staircase", "ledge" — and it will select the
matching scene.

| environment | scene | good for |
|---|---|---|
| `flat` | `flat.xml` | locomotion, acrobatics, expressive motion |
| `carry_box` | `carry.xml` | pick / carry / place (a **mocap** crate) |
| `chair` | `chair.xml` | sitting, stepping up |
| `stairs` | `stairs.xml` | step-by-step climbing |
| `ledge` | `slab.xml` | mantling, vaulting, drops |

Three things make these harder than flat-ground skills, and all three are worth stating:

**Reach.** To touch something low, the robot must get its limb *root* near it — which
means crouching, or pitching the torso forward, not just reaching. The planner is given
a per-limb reach envelope and is checked against it before any code is written, but
saying "crouches all the way down" removes the ambiguity.

**Grip.** An object is only carried if a limb genuinely arrives at it. The framework
**refuses** to snap an object to a limb that missed (it would manufacture impossible
training data), so a missed grip shows up as the object staying put.

**Contacts must alternate.** A walk declares each foot as a contact only while planted.
A skill that declares a static support set while translating is gliding, not walking.
Describing locomotion as *stepping* rather than *moving* helps.

---

## Reading the result

```
skill_output/<robot>/<SKILL>/
├── <SKILL>.py            # the winning code
├── <SKILL>_plan.json     # the motion plan (phases, contacts, channels, invariants)
├── <SKILL>_prompt.md     # the exact prompt that produced it
├── <SKILL>.pkl           # trajectory for evaluation
├── <SKILL>_mimickit.pkl  # for downstream imitation learning
└── <SKILL>.gif           # preview
```

Score every shipped skill against the same evaluator the pipeline uses:

```bash
python agent/score_skills.py --json before.json
python agent/score_skills.py --compare before.json
```

**Watch the GIF.** The score measures physics, not intent — a skill can pass the floor
while doing the wrong thing (a "sit" that squats beside the chair at seat height passes
every physical law). Your eye is the check for intent; the score is the check for physics.

---

## Run-to-run variance is large

The pipeline is non-deterministic and the spread is wider than most differences you will
try to measure. One skill scored **−9.50, −5.50, −4.50, −2.50 and 0.00 on five runs of an
identical prompt**.

Practical consequences:

- **A bad result is worth one re-run** before you conclude anything about the prompt.
- **Don't compare single runs** to judge a change — the difference is usually noise.
- **Keep the better version.** Back up a good skill directory before regenerating over it.

---

## Bypassing the feasibility gate

`skill_filter` vetoes skills it judges infeasible, which can block legitimate aerial or
expressive requests:

```bash
SKIP_SKILL_FILTER=1 python run.py --skill "..." --name SKILL_X
```

---

## See also

- [`ADDING_A_ROBOT.md`](ADDING_A_ROBOT.md) — bringing a new robot in
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the checks, the repair loop, why they are shaped this way
