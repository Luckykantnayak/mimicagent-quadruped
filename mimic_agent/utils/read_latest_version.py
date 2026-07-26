import os
import re

def get_latest_skill_file(base_dir, skill_version, skill_name):
    """
    Returns path to latest <skill_name>_XXXX.json file.
    Raises FileNotFoundError if none found.
    """
    dir_path = os.path.join(base_dir, skill_version, skill_name)

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pattern = re.compile(rf"^{re.escape(skill_name)}_(\d+)\.json$")

    latest_num = -1
    latest_file = None

    for fname in os.listdir(dir_path):
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            if num > latest_num:
                latest_num = num
                latest_file = fname

    if latest_file is None:
        raise FileNotFoundError(
            f"No files found for skill '{skill_name}' in {dir_path}"
        )

    return os.path.join(dir_path, latest_file)
