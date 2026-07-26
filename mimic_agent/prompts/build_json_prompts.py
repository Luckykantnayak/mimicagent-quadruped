import os
import json
import yaml


def load_yaml_config(yaml_path="config.yaml"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = os.path.dirname(os.path.dirname(os.path.abspath(yaml_path)))
    for key, value in cfg.get("paths", {}).items():
        if not os.path.isabs(value):
            cfg["paths"][key] = os.path.normpath(os.path.join(root, value))

    if "robot" in cfg and "urdf" in cfg["robot"]:
        urdf = cfg["robot"]["urdf"]
        if not os.path.isabs(urdf):
            cfg["robot"]["urdf"] = os.path.normpath(os.path.join(root, urdf))

    return cfg


def load_urdf_text(cfg):
    urdf_path = cfg["robot"]["urdf"]
    if os.path.isfile(urdf_path):
        with open(urdf_path, "r", encoding="utf-8") as f:
            return f.read()
    return f"<URDF/MJCF not found at {urdf_path}>"


def load_prompt(name, cfg):
    robot = cfg["robot"]["name"]
    prompt_path = os.path.join(cfg["paths"]["prompts_dir"], robot, f"{name}.json")
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    urdf = load_urdf_text(cfg)
    for key in ("system", "human"):
        data[key] = data[key].replace("{robot}", robot).replace("{urdf}", urdf)

    return data


def convert_txt_prompts_to_json(prompts_subdir):
    txt_files = [f for f in os.listdir(prompts_subdir) if f.endswith(".txt")]

    for txt_file in txt_files:
        txt_path = os.path.join(prompts_subdir, txt_file)
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into system and human
        if "=== HUMAN ===" in content:
            system_prompt, human_prompt = content.split("=== HUMAN ===", 1)
        else:
            system_prompt = content
            human_prompt = ""

        system_prompt = system_prompt.split("=== SYSTEM ===")[1]

        json_data = {
            "system": system_prompt.strip(),
            "human": human_prompt.strip()
        }

        # Save JSON named after txt file
        json_filename = os.path.splitext(txt_file)[0] + ".json"
        json_path = os.path.join(prompts_subdir, json_filename)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2)

        print(f"Saved {json_path}")


if __name__ == "__main__":
    cfg = load_yaml_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg", "config.yaml"))
    prompts_dir = cfg["paths"]["prompts_dir"]
    for robot in sorted(os.listdir(prompts_dir)):
        subdir = os.path.join(prompts_dir, robot)
        if os.path.isdir(subdir):
            convert_txt_prompts_to_json(subdir)
