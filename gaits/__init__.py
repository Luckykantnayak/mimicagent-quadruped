# gaits/__init__.py
"""
Gait registry and dynamic skill loader.

Directory structure expected:
    gaits/
        <skill_name>/
            <skill_name>_1.py
            <skill_name>_2.py   ← latest version is loaded automatically
            ...

Any file matching <skill_name>_<N>.py with a concrete class ending in
'MotionGenerator' will be discovered and returned.

Usage:
    MotionClass = get_motion_controller("humanoid_static", gaits_dir="gaits")
    gen = MotionClass(initial_foot_positions_body=..., leg_names=...)
"""

import os
import sys
import re
import inspect
import importlib.util


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def get_motion_controller(skill_name: str, gaits_dir: str):
    """
    Load and return the latest version of a skill's MotionGenerator class.

    Args:
        skill_name : str   e.g. "static", "humanoid_static", "trot"
        gaits_dir  : str   path to the gaits directory

    Returns:
        A concrete subclass of BaseMotionGenerator (not instantiated).

    Raises:
        FileNotFoundError : if no versioned file is found for the skill
        RuntimeError      : if no concrete MotionGenerator class is found
    """
    skill_file = _find_latest_skill_file(gaits_dir, skill_name)
    module     = _import_module_from_path(f"gaits.{skill_name}_latest", skill_file)
    return _get_motion_generator_from_module(module)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_latest_skill_file(gaits_dir: str, skill_name: str) -> str:
    """
    Scan <gaits_dir>/<skill_name>/ for files matching <skill_name>_<N>.py
    and return the path of the highest-versioned one.
    """
    skill_dir = os.path.join(gaits_dir, skill_name)

    if not os.path.isdir(skill_dir):
        raise FileNotFoundError(
            f"Skill directory not found: '{skill_dir}'\n"
            f"Expected structure: {gaits_dir}/{skill_name}/{skill_name}_1.py"
        )

    pattern  = re.compile(rf"^{re.escape(skill_name)}_(\d+)\.py$")
    versions = []

    for fname in os.listdir(skill_dir):
        m = pattern.match(fname)
        if m:
            versions.append((int(m.group(1)), fname))

    if not versions:
        raise FileNotFoundError(
            f"No versioned files found for skill '{skill_name}' in '{skill_dir}'.\n"
            f"Expected files like: {skill_name}_1.py, {skill_name}_2.py, ..."
        )

    versions.sort()
    latest_file = versions[-1][1]
    print(f"[gaits] Loading skill '{skill_name}' v{versions[-1][0]} → {latest_file}")
    return os.path.join(skill_dir, latest_file)


def _import_module_from_path(module_name: str, file_path: str):
    """Dynamically import a Python file as a module."""
    spec   = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get_motion_generator_from_module(module):
    """
    Find and return the first concrete MotionGenerator class in the module.

    A valid class must:
        - Have a name ending in 'MotionGenerator'
        - Not be abstract
        - Be defined in this module (not imported from elsewhere)
    """
    candidates = []

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            obj.__name__.endswith("MotionGenerator")
            and not inspect.isabstract(obj)
            and obj.__module__ == module.__name__
        ):
            candidates.append(obj)

    if not candidates:
        raise RuntimeError(
            f"No concrete MotionGenerator found in '{module.__file__}'.\n"
            f"Make sure your class name ends with 'MotionGenerator' and is not abstract."
        )

    if len(candidates) > 1:
        print(f"[gaits] WARNING: Multiple MotionGenerators found: "
              f"{[c.__name__ for c in candidates]}. Using first: {candidates[0].__name__}")

    return candidates[0]


# ---------------------------------------------------------------------------
# Optional: list available skills
# ---------------------------------------------------------------------------

def list_available_skills(gaits_dir: str) -> list:
    """
    Return a list of skill names that have at least one versioned file.

    Args:
        gaits_dir : str  path to the gaits directory

    Returns:
        list[str]  e.g. ["static", "trot", "humanoid_static"]
    """
    skills = []

    if not os.path.isdir(gaits_dir):
        return skills

    for entry in os.scandir(gaits_dir):
        if not entry.is_dir():
            continue
        skill_name = entry.name
        pattern    = re.compile(rf"^{re.escape(skill_name)}_(\d+)\.py$")
        has_files  = any(pattern.match(f) for f in os.listdir(entry.path))
        if has_files:
            skills.append(skill_name)

    return sorted(skills)