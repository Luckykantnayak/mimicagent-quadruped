"""
utils/robot_config.py

Loads per-robot YAML configs and provides leg-name normalization so that
all gait files can use the canonical FL/FR/RL/RR convention regardless of
what the underlying robot XML actually calls its legs.

Canonical convention
--------------------
  FL = Front-Left
  FR = Front-Right
  RL = Rear-Left
  RR = Rear-Right

Any robot whose XML uses a different scheme (e.g. ANYmal: LF/RF/LH/RH)
is mapped to/from canonical via normalize_leg_name / denormalize_leg_name.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


# ---------------------------------------------------------------------------
# Leg-name normalization tables
# ---------------------------------------------------------------------------
# Each entry: (robot_prefix, canonical_prefix)
# Add new robots here as needed.
_LEG_PREFIX_MAP: List[tuple[str, str]] = [
    # ANYmal  (LF/RF/LH/RH  →  FL/FR/RL/RR)
    ("LF", "FL"),
    ("RF", "FR"),
    ("LH", "RL"),
    ("RH", "RR"),
    # Spot (fl_/fr_/hl_/hr_ lowercase → FL/FR/RL/RR)
    ("fl_", "FL_"),
    ("fr_", "FR_"),
    ("hl_", "RL_"),
    ("hr_", "RR_"),
    # Barkour VB (front_left / hind_right -> FL / RR)
    ("front_left", "FL"),
    ("front_right", "FR"),
    ("hind_left", "RL"),
    ("hind_right", "RR"),
    # Ant
    ("back_left", "RL"),
    ("back_right", "RR"),
]

# Reverse map built once at module load
_LEG_PREFIX_REVERSE: Dict[str, str] = {
    canonical: robot for robot, canonical in _LEG_PREFIX_MAP
}


def normalize_leg_name(name: str) -> str:
    """
    Translate a robot-specific leg name to the canonical FL/FR/RL/RR scheme.

    Examples
    --------
    >>> normalize_leg_name("LF_shank")   # ANYmal
    'FL_shank'
    >>> normalize_leg_name("FL_calf")    # Go2W — unchanged
    'FL_calf'
    """
    for robot_prefix, canonical_prefix in _LEG_PREFIX_MAP:
        if name.startswith(robot_prefix):
            return canonical_prefix + name[len(robot_prefix):]
    return name  # already canonical


def denormalize_leg_name(canonical: str, robot_name: str) -> str:
    """
    Translate a canonical FL/FR/RL/RR leg name back to the robot-specific form.

    Parameters
    ----------
    canonical  : name in canonical form, e.g. "FL_shank"
    robot_name : robot identifier from config, e.g. "anymal"

    Examples
    --------
    >>> denormalize_leg_name("FL_shank", "anymal")
    'LF_shank'
    >>> denormalize_leg_name("FL_calf", "unitree_go2w")
    'FL_calf'
    """
    robot_prefix_map = _get_robot_prefix_map(robot_name)
    for canonical_prefix, robot_prefix in robot_prefix_map.items():
        if canonical.startswith(canonical_prefix):
            return robot_prefix + canonical[len(canonical_prefix):]
    return canonical  # no mapping needed


def _get_robot_prefix_map(robot_name: str) -> Dict[str, str]:
    """Return canonical→robot prefix mapping for a given robot."""
    rn = robot_name.lower()

    # ANYmal C: Uses underscores and HAA/HFE/KFE naming
    if rn == "anymal_c":
        return {
            "FL_": "LF_",
            "FR_": "RF_",
            "RL_": "LH_",  # Canonical Rear-Left is ANYmal Left-Hind
            "RR_": "RH_"  # Canonical Rear-Right is ANYmal Right-Hind
        }

    # Original ANYmal: Often uses LF/RF/LH/RH without the underscore
    # or different joint suffixes like HAA/HFE/KFE vs hip/thigh/shank
    if rn == "anymal":
        return {"FL": "LF", "FR": "RF", "RL": "LH", "RR": "RH"}

    if "spot" in rn:
        return {"FL_": "fl_", "FR_": "fr_", "RL_": "hl_", "RR_": "hr_"}

    if "barkour" in rn:
        return {
            "FL": "front_left",
            "FR": "front_right",
            "RL": "hind_left",
            "RR": "hind_right"
        }

    if "humanoid" in rn:
        return {}

    if "ant" in rn:
        return {
            "FL": "front_left",
            "FR": "front_right",
            "RL": "back_left",
            "RR": "back_right",
        }
        return {}

    return {}


# ---------------------------------------------------------------------------
# RobotConfig dataclass
# ---------------------------------------------------------------------------
@dataclass
class RobotConfig:
    name: str
    scene_xml: str
    base_body: str
    calf_bodies: List[str]
    foot_sites: List[str]
    joint_names: List[str]
    leg_order: List[str]
    hand_bodies: List[str] = field(default_factory=list)
    hand_sites: List[str] = field(default_factory=list)   # ← add this
    wheel_radius: float = 0.0
    extra: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def canonical_leg_names(self) -> List[str]:
        """Calf body names mapped to canonical FL/FR/RL/RR form."""
        return [normalize_leg_name(n) for n in self.calf_bodies]

    def to_xml_leg_name(self, canonical: str) -> str:
        """Convert canonical leg name to XML leg name for this robot."""
        return denormalize_leg_name(canonical, self.name)

    def to_canonical_leg_name(self, xml_name: str) -> str:
        """Convert XML leg name to canonical leg name."""
        return normalize_leg_name(xml_name)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------
def load_robot_config(robot_name: str, robots_dir: str = "robots") -> RobotConfig:
    """
    Load a robot config from  <robots_dir>/<robot_name>.yaml

    Expected YAML structure
    -----------------------
    name: unitree_go2w
    scene_xml: assets/unitree_go2w/scene.xml
    base_body: base
    calf_bodies: [FL_calf, FR_calf, RL_calf, RR_calf]
    foot_sites:  [FL_foot, FR_foot, RL_foot, RR_foot]
    joint_names: [FL_hip_joint, FL_thigh_joint, ...]
    leg_order:   [FL, FR, RL, RR]
    wheel_radius: 0.1040071605
    """
    path = Path(robots_dir) / f"{robot_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Robot config not found: {path}\n"
            f"Available robots: {[p.stem for p in Path(robots_dir).glob('*.yaml')]}"
        )

    with open(path) as f:
        data = yaml.safe_load(f)

    return RobotConfig(
        name=data["name"],
        scene_xml=data["scene_xml"],
        base_body=data["base_body"],
        calf_bodies=data["calf_bodies"],
        foot_sites=data["foot_sites"],
        joint_names=data["joint_names"],
        leg_order=data.get("leg_order", []),
        hand_bodies=data.get("hand_bodies", []),
        hand_sites=data.get("hand_sites", []),  # ← add this
        wheel_radius=data.get("wheel_radius", 0.0),
        extra=data.get("extra", {}),
    )



config = load_robot_config("anymal_c")
test_joint = "FL_HFE"
xml_joint = config.to_xml_leg_name(test_joint)

print(f"Canonical: {test_joint} -> XML Name: {xml_joint}")
# Expected Output: Canonical: FL_HFE -> XML Name: LF_HFE