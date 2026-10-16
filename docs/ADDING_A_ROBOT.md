# Adding a Robot

[README](../README.md) · [Architecture](ARCHITECTURE.md)

This implementation assumes a quadruped with three controlled joints per leg.
Adding a model requires matching its XML, robot specification, configuration,
and evaluator assumptions. The robot YAML alone does not configure every check.

## 1. Add the model and scene

Place the robot XML and meshes in `assets/robots/<name>/`. Add a scene that
includes the model and provides the floor and lighting:

```xml
<mujoco model="quadruped scene">
  <include file="robot.xml"/>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane"/>
  </worldbody>
</mujoco>
```

The model must provide:

- A named base body with a free joint occupying `qpos[0:7]`.
- Four leg chains with valid joint axes, ranges, and consistent names.
- One foot site per leg, used as its IK target.
- A named `home` keyframe containing a valid nominal standing pose.

Sites are positioned relative to their parent bodies. A position inherited
through an XML default class also applies in that local frame. Do not repeat
an offset already represented by the parent body: this previously displaced
Go2-W's foot sites below its wheels.

For a point foot, put the site at the contact point. For a wheel, choose a
documented hub-to-ground offset and verify it against the geometry. The current
Go2-W checker uses 0.0801 m, calibrated from its home-pose site height; this is not
a generic wheel-radius measurement.

## 2. Define the robot specification

Create `robots/<name>/robot.yaml`, following the bundled
[Go2-W specification](../robots/unitree_go2w/robot.yaml):

```yaml
name: my_quadruped
scene_xml: assets/robots/my_quadruped/scene.xml
base_body: base_link
calf_bodies: [FL_calf, FR_calf, RL_calf, RR_calf]
foot_sites: [FL_foot, FR_foot, RL_foot, RR_foot]
joint_names:
  - FL_hip_joint
  - FL_thigh_joint
  - FL_calf_joint
  - FR_hip_joint
  - FR_thigh_joint
  - FR_calf_joint
  - RL_hip_joint
  - RL_thigh_joint
  - RL_calf_joint
  - RR_hip_joint
  - RR_thigh_joint
  - RR_calf_joint
leg_order: [FL, FR, RL, RR]
wheel_radius: 0.0
```

Every body, site, and joint name must exist in the model. Keep the leg lists
aligned in the same order. Verify the actual model joint layout, including any
passive wheel joints, against the controller and evaluator indexing.

Known naming conventions are mapped in
[robot_config.py](../simulator/utils/robot_config.py). Add both normalization and
reverse mappings for a new convention. Normalized leg names preserve suffixes:
`LF_shank` becomes `FL_shank`, not bare `FL`.

## 3. Select it in both configurations

```yaml
# simulator/config.yaml
run:
  robot: my_quadruped

# agent/cfg/config.yaml
robot:
  name: my_quadruped
  urdf: ../assets/robots/my_quadruped/robot.xml
```

Despite the `urdf` key name, the bundled setup passes MJCF XML text to the prompts.
Also change the agent's robot-specific output paths (`gait_dir`,
`data_output_dir`, `motion_planner_output_dir`, `results_dir`, `metadata_dir`,
and `trajectory_dir`) to the new robot's output directory.

## 4. Adapt evaluation assumptions

Review these before accepting generated skills:

- [run_unit_test.py](../agent/agents/code_gen/run_unit_test.py): joint-limit values
  and ordering.
- [test_motion_quality.py](../agent/utils/test_motion_quality.py): passive wheel
  indices, four-leg reshaping, base-height bounds, and foot/contact offset.
- [controller.py](../simulator/core/controller.py) and
  [kinematics.py](../simulator/utils/kinematics.py): chain selection and joint indexing.

Changing `wheel_radius` in YAML does not update the scorer's hardcoded
`WHEEL_RADIUS`. Point-foot robots need the scorer's contact offset adjusted too.

## 5. Check geometry before generation

Load the scene, reset to `home`, and inspect each configured foot site:

```python
import mujoco

model = mujoco.MjModel.from_xml_path("assets/robots/my_quadruped/scene.xml")
data = mujoco.MjData(model)
home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
assert home >= 0, "Missing home keyframe"
mujoco.mj_resetDataKeyframe(model, data, home)
mujoco.mj_fwdPosition(model, data)

for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    assert site >= 0, f"Missing site: {name}"
    print(name, data.site_xpos[site])
```

Verify site heights against the floor and intended contact offset, legal home
joint angles, and leg naming/order. Then replay a simple hold-pose skill using
the [motion interface](ARCHITECTURE.md#motion-interface) before generating more
complex motions. A kinematic replay does not validate balance or hardware safety.
