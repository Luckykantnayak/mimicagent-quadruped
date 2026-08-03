# MimicAgent — Text-to-Motion Skill Generation for Humanoids

#### Given a natural-language skill description, an LLM agent writes the motion-generator code, executes it in a **MuJoCo** kinematic simulator, checks the trajectory against physical laws *and* against fidelity invariants the agent itself declared, then repairs its own code until the motion is both legal and recognizably the thing that was asked for. Trajectories export in **MimicKit-compatible** `.pkl` format for downstream imitation learning (DeepMimic and friends).

---

## Features

- Natural language → executable motion-generator code → recorded trajectory
- Morphology-agnostic: one spec file (`robots/<name>/robot.yaml`) drives both the simulator and every prompt
- Robot-agnostic scenes — floor, stairs, ledge, chair, carry-box — composed with any robot at load
- Three control channels (world-IK / body-IK / joint), chosen per limb per phase by the agent
- Physics floor + agent-declared fidelity invariants, with an automatic repair loop
- Pre-sim geometry lint: rejects unbuildable plans in milliseconds instead of a full rollout
- Pure kinematic IK with bounded, recorded safety nets — no physics solve
- Exports MimicKit `.pkl` + preview GIF per skill

---

## 1. Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="..."
```

Ships one robot: a **28-DoF humanoid**, bundled under `assets/robots/humanoid/`.
Set the LLM model and target robot in `agent/cfg/config.yaml`.

> Run every command **from the repo root**.

## 2. Play the reference skill

```bash
python simulator/run.py --robot humanoid --record True
```

## 3. Generate a skill

```bash
cd agent
python run.py --skill "The humanoid walks forward at a steady pace" --name SKILL_WALK
```

Output lands in `skill_output/humanoid/SKILL_WALK/` — the winning code, its motion
plan, the exact prompt, `.pkl` trajectories, and a preview GIF.

Prompt phrasing matters more than anything else for motion quality —
see [`docs/SKILL_GUIDE.md`](docs/SKILL_GUIDE.md).

## 4. Play a generated skill

```bash
python simulator/run.py --robot humanoid --mode SKILL_WALK --record True
```

`--mode` reads the skill's own scene, grounding and repetitions from its motion plan,
so a bare replay reproduces the agent's recording.

## 5. Score skills

```bash
python agent/score_skills.py --json before.json
python agent/score_skills.py --compare before.json
```

## 6. Interactive demo (Streamlit)

```bash
cd agent
streamlit run app.py
```

## 7. Add a robot

```bash
python simulator/tools/gen_robot_spec.py    --robot <name>   # scaffold robot.yaml (review the # REVIEW marks)
python simulator/tools/gen_robot_summary.py --robot <name>   # summary.md + dynamics.json
python simulator/run.py --robot <name>                       # plays the example
```

Assets + one spec file + one reference skill. No prompt or code changes.
Full walkthrough — MJCF requirements, every `robot.yaml` field, constants to retune —
in [`docs/ADDING_A_ROBOT.md`](docs/ADDING_A_ROBOT.md).

---

## Layout

```
robots/<name>/      robot.yaml (spec), example.py, exemplars/, summary.md, dynamics.json
assets/             robots/<name>/ (MuJoCo model + meshes), scenes/ (robot-agnostic)
simulator/          MuJoCo kinematic simulator — run.py, core/, motion/, utils/, tools/
agent/              LLM pipeline — skill → plan → code → trajectory → repair
docs/               documentation (below)
skill_output/       generated skills + artifacts (gitignored)
```

## Documentation

| doc | what it covers |
|---|---|
| [`docs/ADDING_A_ROBOT.md`](docs/ADDING_A_ROBOT.md) | MJCF format, `robot.yaml` reference, `example.py` contract, scale-dependent constants, verification checklist |
| [`docs/SKILL_GUIDE.md`](docs/SKILL_GUIDE.md) | writing skill prompts, scene-interaction guidelines, reading results |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | simulator internals, the skill contract, the checks, the repair loop |

## Environment toggles

`SKIP_SKILL_FILTER=1` bypasses the feasibility gate · `ROBOT_NAME=<name>` overrides the
target robot per process · simulator nets `REACH_PROJ` `SEAM_BRIDGE` `GROUNDING_CLAMP`
`RATE_LIMIT` `CONTINUITY_REF` `EE_ORIENT` (`=0` to disable) · `NO_DRAG=1` opts in to the
contact-drag lint · `IK_BACKEND=dls|mink`.

## License

See `LICENSE`.
