# from gaits.walking import WalkingMotionGenerator
# from gaits.skating import SkatingMotionGenerator
# from gaits.static import StaticMotionGenerator
#
# # gaits/__init__.py
# GAIT_REGISTRY = {
#     "walking": WalkingMotionGenerator,
#     "skating": SkatingMotionGenerator,
#     "static": StaticMotionGenerator
#
# }
#
# def get_motion_controller(name):
#     return GAIT_REGISTRY[name]  # <-- return the class, do not call it
#
import os
import pkgutil
import importlib
import re
import sys
import inspect

import importlib.util


GAIT_REGISTRY = {}

def _auto_register_gaits():
    import gaits

    for module_info in pkgutil.iter_modules(gaits.__path__):
        module = importlib.import_module(f"gaits.{module_info.name}")

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                if obj.__name__.endswith("MotionGenerator"):
                    name = module_info.name
                    GAIT_REGISTRY[name] = obj

_auto_register_gaits()

# def get_motion_controller(name):
#     return GAIT_REGISTRY[name]

def get_motion_controller(skill_name:str, gaits_dir:str):
    skill_file = find_latest_skill_file(gaits_dir, skill_name)
    module_name = f"gaits.{skill_name}_latest"

    module = import_module_from_path(module_name, skill_file)
    MotionGenerator = get_motion_generator_from_module(module)

    return MotionGenerator


def find_latest_skill_file(gaits_dir:str, skill_name:str):
    gaits_dir = os.path.join(gaits_dir, skill_name)
    pattern = re.compile(rf"{skill_name}_(\d+)\.py$")
    versions = []

    for fname in os.listdir(gaits_dir):
        m = pattern.match(fname)
        if m:
            versions.append((int(m.group(1)), fname))

    if not versions:
        raise FileNotFoundError(f"No versions found for skill '{skill_name}'")

    versions.sort()
    return os.path.join(gaits_dir, versions[-1][1])


def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_motion_generator_from_module(module):
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            obj.__name__.endswith("MotionGenerator")
            and not inspect.isabstract(obj)
        ):
            return obj
    raise RuntimeError("No concrete MotionGenerator found")

