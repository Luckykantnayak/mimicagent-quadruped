import pickle
from pathlib import Path
import yaml

def save_pickle(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Saved file: {path}")

def load_pickle(path):
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)

def load_yaml(path):
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    print(f"✅ Saved YAML: {path}")
