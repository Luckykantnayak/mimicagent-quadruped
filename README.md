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

## Adding another quadruped

Three things must agree: the MuJoCo model, the robot spec that indexes it, and the two
config files that select it. Nothing else in the pipeline is robot-specific.

### 1. The MuJoCo model — `assets/robots/<name>/`

Ship the model and its meshes here, plus a `scene.xml` that includes the robot and adds
the world:

```xml
<mujoco model="<name> scene">
  <include file="<name>.xml"/>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane"/>
  </worldbody>
</mujoco>
```

The robot XML must provide four things:

**A free-floating base body.** One named body the whole kinematic tree hangs from, with a
free joint, so `qpos[0:3]` is its position and `qpos[3:7]` its `[w,x,y,z]` orientation —
the simulator writes the skill's root pose straight into those slots.

**Four legs with three actuated joints each**, in a consistent naming scheme. Any prefix
convention works (`FL/FR/RL/RR`, ANYmal's `LF/RF/LH/RH`, Spot's lowercase `fl_/fr_/hl_/hr_`);
`simulator/utils/robot_config.py` normalises the known ones to canonical `FL/FR/RL/RR`, and
generated skills only ever see the canonical names. Add a mapping there for an unknown scheme.

**One site per foot.** These are the IK targets — the single most important detail:

> The site marks the point the solver drives to a commanded position. Put it at the
> **ground-contact point**, or at a fixed known offset above it (a wheel hub). Then set
> `wheel_radius` to the distance from the site down to the contact point, because the
> quality checker computes `contact_z = site_z - wheel_radius`.
>
> Watch for double-counted offsets. If a site is declared through a `<default class>` that
> carries a `pos`, that offset is applied **relative to its parent body** — and if the
> parent is already displaced down the chain, the offset lands twice. In this repo's Go2-W
> that put every site 0.184 m below its wheel, so the IK drove a point 18 cm underground
> and the ground checks silently compensated for it. Verify with forward kinematics before
> trusting it (see the sanity check below).

**A `home` keyframe** giving a valid standing pose with the feet on the floor. It is the
reset state, and `initial_foot_positions_body` — the nominal stance every skill builds
from — is measured from it.

### 2. The robot spec — `robots/<name>/robot.yaml`

An index into the XML. Every name must match the model exactly:

```yaml
name: <name>                                   # must equal the directory name
scene_xml: assets/robots/<name>/scene.xml      # repo-root-relative
base_body: base_link                           # the free-floating root body
calf_bodies:  [FL_calf, FR_calf, RL_calf, RR_calf]     # one per leg, IK chain tips
foot_sites:   [FL_foot, FR_foot, RL_foot, RR_foot]     # same order as calf_bodies
joint_names:  [FL_hip_joint, FL_thigh_joint, FL_calf_joint,   # qpos[7:] order
               FR_hip_joint, FR_thigh_joint, FR_calf_joint,
               RL_hip_joint, RL_thigh_joint, RL_calf_joint,
               RR_hip_joint, RR_thigh_joint, RR_calf_joint]
leg_order:    [FL, FR, RL, RR]
wheel_radius: 0.0801                            # site -> contact distance; 0 for a point foot
```

`joint_names` must be in **model `qpos[7:]` order** — the recorder and the joint-limit
check both index positionally, so a mismatch silently scores the wrong joint. Legged
robots set `wheel_radius: 0.0` and place the sites at the sole.

### 3. Select it

```yaml
# simulator/config.yaml
run:
  robot: <name>          # which robot the simulator loads

# agent/cfg/config.yaml
robot:
  name: <name>           # fills {robot} in the prompts
  urdf: ../assets/robots/<name>/<name>.xml   # fills {urdf} — the model text the coder reads
```

These are deliberately separate and **must be changed together**: the agent's subprocess
call does not pass `--robot`, so the simulated robot always comes from the simulator config.

### 4. Sanity-check the geometry before generating anything

Confirm the feet land where the spec claims. A wrong site offset costs nothing to fix now
and invalidates every skill later:

```python
import mujoco
m = mujoco.MjModel.from_xml_path("assets/robots/<name>/scene.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
mujoco.mj_fwdPosition(m, d)
for leg in ["FL", "FR", "RL", "RR"]:
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot")
    print(leg, d.site_xpos[sid])          # z should equal wheel_radius (0 for a point foot)
```

Then play the bundled reference skill to see it move:

```bash
python simulator/run.py --mode SKILL_TROT --record True --version check
```

The joint limits in `agent/agents/code_gen/run_unit_test.py` and the base-height envelope
in `agent/utils/test_motion_quality.py` are still Go2-W values — update both for a robot
of a different size, or every skill will be scored against the wrong bounds.

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
