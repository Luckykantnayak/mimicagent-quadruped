# core/config.py
import argparse
from utils.file_utils import load_yaml
from pathlib import Path


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "1"): return True
    if v.lower() in ("no", "false", "f", "0"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_config(default_config_path="config/config.yaml"):
    default_config_path = Path(__file__).parent.parent / "config/config.yaml"
    cfg = load_yaml(default_config_path)

    parser = argparse.ArgumentParser()

    # Run block
    parser.add_argument("--mode",           type=str,      default=cfg["run"]["mode"])
    parser.add_argument("--robot",          type=str,      default=cfg["run"]["robot"])
    # NEW: robots_dir lets you point to wherever your robot YAMLs live
    parser.add_argument("--robots_dir",     type=str,      default=cfg["run"].get("robots_dir", "robots"))
    parser.add_argument("--record",         type=str2bool, default=cfg["run"]["record"])
    parser.add_argument("--plot",           type=str2bool, default=cfg["run"]["plot"])
    parser.add_argument("--record_cycles",  type=int,      default=cfg["run"]["record_cycles"])
    parser.add_argument("--init_position",  type=str,      default=cfg["run"]["init_position"])
    parser.add_argument("--sim_freq",       type=float,    default=cfg["run"]["sim_freq"])
    parser.add_argument("--version",        type=str,      default=cfg["run"]["version"])

    # Motion block
    parser.add_argument("--vx",         type=float, default=cfg["motion"]["vx"])
    parser.add_argument("--vy",         type=float, default=cfg["motion"]["vy"])
    parser.add_argument("--vz",         type=float, default=cfg["motion"]["vz"])
    parser.add_argument("--roll_rate",  type=float, default=cfg["motion"]["roll_rate"])
    parser.add_argument("--pitch_rate", type=float, default=cfg["motion"]["pitch_rate"])
    parser.add_argument("--yaw_rate",   type=float, default=cfg["motion"]["yaw_rate"])

    # Gait block
    parser.add_argument("--gait_freq",    type=float, default=cfg["gait"]["freq"])
    parser.add_argument("--step_length",  type=float, default=cfg["gait"]["params"]["step_length"])
    parser.add_argument("--step_height",  type=float, default=cfg["gait"]["params"]["step_height"])
    parser.add_argument("--duty_ratio",   type=float, default=cfg["gait"]["params"]["duty_ratio"])

    # Skating block
    parser.add_argument("--skating_style", type=str, default=cfg["skating"]["style"])

    return parser.parse_args()