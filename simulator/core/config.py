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
    parser.add_argument("--record_cycles",  type=int,      default=cfg["run"]["record_cycles"])
    parser.add_argument("--init_position",  type=str,      default=cfg["run"]["init_position"])
    parser.add_argument("--sim_freq",       type=float,    default=cfg["run"]["sim_freq"])

    # Gait block (recording length only; each skill owns its own motion)
    parser.add_argument("--gait_freq",    type=float, default=cfg["gait"]["freq"])

    return parser.parse_args()