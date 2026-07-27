# MimicAgent — quadruped skills from natural language

Describe a motion in plain English. An LLM agent plans it, writes a Python motion
generator, runs it in a kinematic MuJoCo simulator, scores the recorded trajectory,
and rewrites its own code until the motion passes.

```
"The robot trots forward at a steady pace"
        ↓
   motion plan → generated code → simulated → scored → repaired if needed
        ↓
   skill_output/unitree_go2w/SKILL_TROT/SKILL_TROT_0002.py   +  trajectory .pkl
```

## Setup

Python 3.10+, and a display (the simulator opens a window).

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

Unitree Go2 and Go2-W MuJoCo models are bundled. No other downloads needed.

## Generate a skill

```bash
cd agent
python run.py --skill "The robot bounds forward like a rabbit" --name SKILL_BOUND
```

Takes a few minutes. Output lands in `skill_output/unitree_go2w/SKILL_BOUND/`:
`SKILL_BOUND_0001.py`, `_0002.py`, … — one file per repair iteration, highest number
is the accepted version.

Batch over the bundled 100-skill library instead:

```bash
cd agent && python run.py
```

## Play a skill

Run from the **repo root**:

```bash
python simulator/run.py --mode SKILL_BOUND --record True --version demo
```

`--mode` is the skill directory name. Recording writes a `.pkl` trajectory under
`skill_output/unitree_go2w/trajectories/<version>/`.

## Browser demo

```bash
cd agent && streamlit run app.py
```

Type a skill and watch each stage stream live.

## Configuration

| File | Sets |
|---|---|
| `agent/cfg/config.yaml` | LLM model, `critic.max_iters` (repair budget), output paths |
| `simulator/config.yaml` | `run.robot`, `sim_freq`, `record_cycles` |

The two `robot` fields are separate: `robot.name` in the agent config selects the
prompt flavour; `run.robot` in the simulator config selects the MuJoCo model. Change
both together.

## Writing a skill by hand

Generated skills subclass `BaseMotionGenerator` (`simulator/motion/base.py`). The
whole contract:

```python
from utils.math_utils import *
from motion.base import BaseMotionGenerator
import numpy as np

class SKILL_X_MotionGenerator(BaseMotionGenerator):
    def __init__(self, initial_foot_positions_body, leg_names=None):
        super().__init__(initial_foot_positions_body, leg_names)
        self.freq = 1.0                      # complete cycles per second

    def compute_foot_position_body_frame(self, leg_name, phase):
        return self.base_feet_pos[leg_name].copy()   # (3,) in BODY frame
```

- Foot targets are **body frame**; base pose is **world frame**; `phase` ∈ [0, 1).
- Make the motion **periodic** — one complete skill per phase cycle. The recorder
  captures whole cycles at whatever `self.freq` you choose.
- Optionally override `compute_phase(t)` and `update_base_motion(phase, dt)`; set
  base motion with the inherited `set_velocity_command` /
  `set_angular_velocity_command`. **Never override `step()` or `reset()`.**
- Legs are always `FL`, `FR`, `RL`, `RR`. Everything is kinematic — no forces or
  contact dynamics.

Drop the file at `skill_output/unitree_go2w/SKILL_X/SKILL_X_0001.py` and play it.

## Quality checks

Every trajectory is scored by `agent/utils/test_motion_quality.py`:

| Check | Rule |
|---|---|
| Base height | world base z within 0.10–0.68 m |
| Ground penetration | `foot_z − 0.0801` ≥ −0.02 for every foot, every frame |
| Joint limits | hip ±1.0472; front thigh −1.5708…3.4907, rear −0.5236…4.5379; calf −2.7227…−0.83776 |
| Airtime | all four feet may not be airborne >0.5 s (only when the motion plan enables it) |

A failure feeds the diagnosis and critic agents, which rewrite the code and retry,
up to `critic.max_iters`.

**These check legality, not fidelity.** A skill that stands still passes everything.
If you need to know the motion is what you asked for, measure the recorded trajectory
yourself — rotation, travel, and height are all in the `.pkl`.

## Adding a robot

1. Put the MuJoCo model in `assets/robots/<name>/` with a `scene.xml` and a `home` keyframe.
2. Write `robots/<name>/robot.yaml` — `scene_xml`, `base_body`, `calf_bodies`,
   `foot_sites`, `joint_names`, `wheel_radius`.
3. Set `run.robot: <name>` in `simulator/config.yaml`.

Foot sites are the IK targets — place them at the ground-contact point or the wheel hub,
and make `wheel_radius` the distance from the site down to the contact point.

## Layout

```
agent/          LLM pipeline — agents/, prompts/templates/, cfg/, run.py
simulator/      MuJoCo simulator — run.py, core/, motion/, utils/
robots/<name>/  per-robot spec (robot.yaml)
assets/robots/  MuJoCo models and meshes
skill_output/   generated skills and trajectories (gitignored)
```

## License

See `LICENSE`.
