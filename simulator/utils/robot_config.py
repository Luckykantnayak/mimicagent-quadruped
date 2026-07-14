"""
simulator/utils/robot_config.py

Loads the per-robot YAML spec — the single source of truth for a robot's
morphology, shared by the simulator and the LLM agent (agent/).

Expected YAML structure
-----------------------
name: unitree_go2w
scene_xml: assets/unitree_go2w/scene.xml
mjcf: assets/unitree_go2w/go2w.xml
base_body: base_link
sim_freq: 200
end_effectors:                       # ordered; names are what gait code uses
  - {name: FL, body: FL_calf, site: FL_foot, ground_offset: 0.104}
  - ...
joints:                              # in model qpos[7:] order
  - {name: FL_hip_joint, limits: [-1.0472, 1.0472]}
  - {name: FL_foot_joint, passive: true}
  - ...
base_height_range: [0.10, 0.68]
camera: {lookat: [0.0, 0.0, 0.35], distance: 2.5, azimuth: 90, elevation: -20}
example_skill: robots/examples/unitree_go2w.py
robot_card: |
  Free-text morphology description injected into the agent prompts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


@dataclass
class RobotSpec:
    name: str
    scene_xml: str
    mjcf: str
    base_body: str
    sim_freq: int
    end_effectors: List[Dict]      # [{name, body, site, ground_offset}, ...]
    joints: List[Dict]             # [{name, limits, passive}, ...] in qpos[7:] order
    base_height_range: Tuple[float, float]
    camera: Dict = field(default_factory=dict)
    example_skill: str = ""
    robot_card: str = ""

    # ------------------------------------------------------------------
    # Derived: end-effectors
    # ------------------------------------------------------------------
    @property
    def ee_names(self) -> List[str]:
        return [ee["name"] for ee in self.end_effectors]

    @property
    def ee_bodies(self) -> List[str]:
        return [ee["body"] for ee in self.end_effectors]

    @property
    def ee_sites(self) -> List[str]:
        return [ee.get("site", ee["body"]) for ee in self.end_effectors]

    @property
    def ee_ground_offsets(self) -> List[float]:
        return [float(ee.get("ground_offset", 0.0)) for ee in self.end_effectors]

    # ------------------------------------------------------------------
    # Derived: joints
    # ------------------------------------------------------------------
    @property
    def joint_names(self) -> List[str]:
        """All joint names, in model qpos[7:] order (what gets recorded)."""
        return [j["name"] for j in self.joints]

    @property
    def actuated_joint_names(self) -> List[str]:
        return [j["name"] for j in self.joints if not j.get("passive", False)]

    @property
    def passive_joint_indices(self) -> List[int]:
        """Indices into the recorded joint vector that are unactuated."""
        return [i for i, j in enumerate(self.joints) if j.get("passive", False)]

    @property
    def joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """Actuated joints only, in order: name -> (lo, hi)."""
        return {
            j["name"]: tuple(j["limits"])
            for j in self.joints
            if not j.get("passive", False)
        }


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------
_REQUIRED_KEYS = [
    "name", "scene_xml", "base_body", "end_effectors", "joints",
    "base_height_range",
]


def load_robot_spec(robot_name: str, robots_dir: str = "robots") -> RobotSpec:
    """Load a robot spec from <robots_dir>/<robot_name>/robot.yaml"""
    path = Path(robots_dir) / robot_name / "robot.yaml"
    if not path.exists():
        available = sorted(
            p.parent.name for p in Path(robots_dir).glob("*/robot.yaml")
        )
        raise FileNotFoundError(
            f"Robot config not found: {path}\n"
            f"Available robots: {available}"
        )

    with open(path) as f:
        data = yaml.safe_load(f)

    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"Robot config {path} missing required keys: {missing}")

    for j in data["joints"]:
        if not j.get("passive", False) and "limits" not in j:
            raise ValueError(
                f"Robot config {path}: joint '{j.get('name')}' needs 'limits' "
                f"(or 'passive: true')"
            )

    return RobotSpec(
        name=data["name"],
        scene_xml=data["scene_xml"],
        mjcf=data.get("mjcf", ""),
        base_body=data["base_body"],
        sim_freq=int(data.get("sim_freq", 200)),
        end_effectors=data["end_effectors"],
        joints=data["joints"],
        base_height_range=tuple(data["base_height_range"]),
        camera=data.get("camera", {}),
        example_skill=data.get("example_skill", ""),
        robot_card=data.get("robot_card", ""),
    )
