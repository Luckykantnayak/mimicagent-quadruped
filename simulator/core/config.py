# simulator/core/config.py
import argparse
from utils.file_utils import load_yaml
from pathlib import Path


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "1"): return True
    if v.lower() in ("no", "false", "f", "0"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_config(default_config_path="config.yaml"):
    default_config_path = Path(__file__).parent.parent / "config.yaml"
    cfg = load_yaml(default_config_path)

    parser = argparse.ArgumentParser()

    # Run block
    parser.add_argument("--mode",           type=str,      default=cfg["run"]["mode"])
    parser.add_argument("--robot",          type=str,      default=cfg["run"]["robot"])
    # robots_dir points at the directory of per-robot dirs (each with robot.yaml)
    parser.add_argument("--robots_dir",     type=str,      default=cfg["run"].get("robots_dir", "robots"))
    parser.add_argument("--record",         type=str2bool, default=cfg["run"]["record"])
    parser.add_argument("--plot",           type=str2bool, default=cfg["run"]["plot"])
    parser.add_argument("--init_position",  type=str,      default=cfg["run"]["init_position"])
    parser.add_argument("--sim_freq",       type=float,    default=cfg["run"]["sim_freq"])
    parser.add_argument("--headless",       type=str2bool, default=cfg["run"].get("headless", False))

    # --- REPLAY-RESOLVED ARGS (default None = "not given on the command line") ---
    # These three describe HOW a skill was recorded, and a generated skill records
    # that in its own motion plan. Defaulting them to None lets run.py apply the
    # precedence  explicit CLI > the skill's plan/exemplar > the config default,
    # so replaying a skill by hand reproduces what the agent recorded instead of
    # silently differing from it. (Recording 2 cycles of a ONE-SHOT skill wraps
    # phase 1 -> 0 and snaps the robot back to its start pose mid-clip.)
    parser.add_argument("--record_cycles",  type=int,      default=None)
    parser.add_argument("--grounding",      type=str,      default=None)
    parser.add_argument("--scene",          type=str,      default=None)

    # Gait block (recording length only; each skill owns its own motion)
    parser.add_argument("--gait_freq",    type=float, default=cfg["gait"]["freq"])

    args = parser.parse_args()
    # Config-file fallbacks for the replay-resolved args, kept separate from the
    # parsed values so run.py can tell "user asked for this" from "nobody said".
    args.cfg_record_cycles = int(cfg["run"].get("record_cycles", 1))
    args.cfg_grounding = cfg["run"].get("grounding", "contact")
    return args