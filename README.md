# MimicAgent: Learning Robot Skills via Text-to-Trajectory Generation

Given a natural-language skill description, an LLM agent writes the code that
generates a motion prior for that skill, executes it in a MuJoCo simulator,
checks the resulting trajectory, and iteratively repairs its own code until the
motion passes. The system is morphology-agnostic — one per-robot spec drives
both the simulator and the agent's prompts.

**Full architecture, pipeline, adding-a-robot guide, and skill-generation
examples: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).**

## Layout

```
.
├── robots/            # one self-contained dir per robot
│   └── <name>/
│       ├── robot.yaml   # spec: end-effectors, joints+limits, camera, robot card
│       ├── example.py   # reference skill (the default demo)
│       └── assets/      # MuJoCo model (scene.xml + model + meshes)
├── simulator/         # the MuJoCo kinematic simulator
│   ├── run.py           # entry point
│   ├── config.yaml
│   ├── core/  motion/  utils/
├── agent/             # the LLM skill-writer (skill -> code -> trajectory, critic loop)
├── skill_output/      # generated skills + artifacts, per robot (gitignored)
└── archive/           # old skills + the input skill library
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="..."
```

MuJoCo models (MimicKit-style) for the Unitree Go2, Go2-W, and a 28-DoF
humanoid are bundled under `robots/<name>/assets/`.

The LLM model (`llm.model`) and target robot (`robot.name`) are set in
`agent/cfg/config.yaml`. Each robot is described by a single spec file
`robots/<name>/robot.yaml` (end-effectors, joints + limits, base-height
envelope, camera, and a free-text "robot card"). A shared prompt template set
in `agent/prompts/templates/` is filled from that spec at load time, so adding
a robot needs no prompt or code changes — just a new `robots/<name>/` directory
(spec + `example.py` + `assets/`). Generated skills are saved under
`skill_output/<robot>/<skill>/`.

## Usage

Run everything **from the repo root**.

Play a robot's reference skill in the simulator:

```bash
python simulator/run.py --robot unitree_go2 --record True
```

Generate a single skill from a natural-language description:

```bash
cd agent
python run.py --skill "The robot trots forward at a steady pace" --name SKILL_TROT
```

Play an agent-generated skill:

```bash
python simulator/run.py --robot unitree_go2 --mode SKILL_TROT --record True
```

### Interactive demo (Streamlit)

Type a skill in the browser and watch each agent's output and status stream live:

```bash
cd agent
streamlit run app.py
```

Enter your Anthropic API key and pick the robot in the sidebar, then describe a
skill and hit **Run**.

## License

See `LICENSE`.
