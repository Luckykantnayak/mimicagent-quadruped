# MimicAgent: Learning Quadruped Skills via Text-to-Trajectory Generation

Given a natural-language skill description, an LLM agent writes the code that
generates a motion prior for that skill, executes it in a MuJoCo simulator,
checks the resulting trajectory, and iteratively repairs its own code until the
motion passes. The repository contains the agent (`agent/`) and the
simulator it drives (`simulator/`).

## Layout

```
.
├── agent/               # the LLM agent (skill -> code -> trajectory, with critic loop)
├── simulator/           # the MuJoCo kinematic simulator
│   ├── run.py           #   entry point
│   ├── config.yaml
│   └── core/  motion/  utils/  tools/
├── robots/<name>/       # per-robot spec (robot.yaml)
├── assets/robots/<name>/  # MuJoCo model + meshes
├── skill_output/<robot>/  # generated skill code + run artifacts
└── skill_library/       # bundled input skill library (all_skills.json)
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="..."
```

The Unitree Go2 and Go2-W MuJoCo models are bundled under `assets/robots/`. The
humanoid model is not included; supply it at `assets/robots/humanoid/` to use
`robots/humanoid/robot.yaml`.

The LLM model (`llm.model`) and target robot (`robot.name`, `robot.urdf`) are set
in `agent/cfg/config.yaml`. Prompts are per-robot under
`agent/prompts/<robot>/`, and the `{robot}` / `{urdf}` fields are filled in
automatically at load time. The robot the *simulator* drives is a separate
setting, `run.robot` in `simulator/config.yaml`.

## Usage

Synthesise motion priors for the whole skill library:

```bash
cd agent
python run.py
```

Or run a single skill straight from a natural-language description:

```bash
cd agent
python run.py --skill "The robot trots forward at a steady pace" --name SKILL_TROT
```

Generate a new skill library (optional):

```bash
cd agent
python run_skill_gen.py
```

Run the simulator on a single skill directly (optional):

```bash
python simulator/run.py --mode <skill_name> --record True --version <skill_version>
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
