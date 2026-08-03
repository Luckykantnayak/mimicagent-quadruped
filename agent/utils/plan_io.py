"""
Bridge to the simulator's motion-plan reader.

`agent/utils` shadows the simulator's `utils` package, so we load
simulator/utils/plan_io.py by file path instead of importing it — the same
pattern as `agent/utils/robot_spec.py`.

The plan is a shared on-disk contract (the agent writes it, the simulator reads
it to replay a skill exactly as recorded), so there is exactly ONE implementation
and it lives simulator-side. Import `load_plan_dict` / `extract_invariants` /
`plan_meta` from here anywhere in the agent.
"""
import importlib.util
import os
import sys

_MODULE_NAME = "sim_plan_io"


def _module():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # agent/
    repo_root = os.path.dirname(here)
    path = os.path.join(repo_root, "simulator", "utils", "plan_io.py")
    spec_ = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec_)
    sys.modules[_MODULE_NAME] = module
    spec_.loader.exec_module(module)
    return module


def load_plan_dict(skill_dir, skill_name=None):
    return _module().load_plan_dict(skill_dir, skill_name)


def extract_invariants(plan):
    return _module().extract_invariants(plan)


def plan_meta(plan):
    return _module().plan_meta(plan)
