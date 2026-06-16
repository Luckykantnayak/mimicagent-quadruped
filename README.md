# MimicAgent: Learning Quadruped Skills via Text-to-Trajectory Generation

Given a natural-language skill description, an LLM agent writes the code that
generates a motion prior for that skill, executes it in a MuJoCo simulator,
checks the resulting trajectory, and iteratively repairs its own code until the
motion passes. The repository contains the agent (`mimic_agent/`) and the
simulator it drives (repo root).

## Layout

```
code/
├── mimic_agent/   # the LLM agent (skill -> code -> trajectory, with critic loop)
├── gaits/         # LLM-generated skill code (the agent's output)
├── core/ assets/ robots/ config/   # the MuJoCo simulator
└── run.py         # simulator entry point
data/              # skill library (all_skills.json); unzip data.zip here, sibling to code/
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="..."
```

Unzip `data.zip` next to this folder so that `code/` and `data/` are siblings.

Robot models are not included. Place the Unitree Go2 / Go2-W MuJoCo models under
`assets/` (e.g. from MuJoCo Menagerie) to run the simulator.

The LLM model (`llm.model`) and target robot (`robot.name`, `robot.urdf`) are set
in `mimic_agent/cfg/config.yaml`. Prompts are per-robot under
`mimic_agent/prompts/<robot>/`, and the `{robot}` / `{urdf}` fields are filled in
automatically at load time.

## Usage

Synthesise motion priors for the whole skill library:

```bash
cd mimic_agent
python run.py
```

Or run a single skill straight from a natural-language description:

```bash
cd mimic_agent
python run.py --skill "The robot trots forward at a steady pace" --name SKILL_TROT
```

Generate a new skill library (optional):

```bash
cd mimic_agent
python run_skill_gen.py
```

Run the simulator on a single skill directly (optional):

```bash
python run.py --mode <skill_name> --record True --version <skill_version>
```

### Interactive demo (Streamlit)

Type a skill in the browser and watch each agent's output and status stream live:

```bash
cd mimic_agent
streamlit run app.py
```

Enter your Anthropic API key and pick the robot in the sidebar, then describe a
skill and hit **Run**.

## License

See `LICENSE`.
