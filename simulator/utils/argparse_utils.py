import argparse
from pathlib import Path
from utils.file_utils import load_yaml

def get_args(default_config_path="config/config.yaml"):
    import argparse
    from utils.file_utils import load_yaml

    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ("yes", "true", "t", "1"): return True
        if v.lower() in ("no", "false", "f", "0"): return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    # ---- Load YAML first ----
    cfg = load_yaml(default_config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=cfg["run"]["mode"])
    parser.add_argument("--robot", type=str, default=cfg["run"]["robot"])
    parser.add_argument("--record", type=str2bool, default=cfg["run"]["record"])
    parser.add_argument("--plot", type=str2bool, default=cfg["run"]["plot"])
    parser.add_argument("--record_cycles", type=int, default=cfg["run"]["record_cycles"])
    parser.add_argument("--init_position", type=str, default=cfg["run"]["init_position"])
    parser.add_argument("--sim_freq", type=float, default=cfg["run"]["sim_freq"])

    parser.add_argument("--base_velocity", type=float, default=cfg["motion"]["base_velocity"])
    parser.add_argument("--yaw_rate", type=float, default=cfg["motion"]["yaw_rate"])

    parser.add_argument("--gait_name", type=str, default=cfg["gait"]["name"])
    parser.add_argument("--gait_freq", type=float, default=cfg["gait"]["freq"])
    parser.add_argument("--step_length", type=float, default=cfg["gait"]["params"]["step_length"])
    parser.add_argument("--step_height", type=float, default=cfg["gait"]["params"]["step_height"])
    parser.add_argument("--duty_ratio", type=float, default=cfg["gait"]["params"]["duty_ratio"])

    parser.add_argument("--skating_style", type=str, default=cfg["skating"]["style"])

    parser.add_argument("--vx", type=float, default=0.0, help="Base linear velocity X (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Base linear velocity Y (m/s)")
    parser.add_argument("--vz", type=float, default=0.0, help="Base linear velocity Z (m/s)")

    return parser.parse_args()

