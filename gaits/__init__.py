from gaits.walking import WalkingMotionGenerator
from gaits.skating import SkatingMotionGenerator
from gaits.static import StaticMotionGenerator

# gaits/__init__.py
GAIT_REGISTRY = {
    "walking": WalkingMotionGenerator,
    "skating": SkatingMotionGenerator,
    "static": StaticMotionGenerator

}

def get_motion_controller(name):
    return GAIT_REGISTRY[name]  # <-- return the class, do not call it

