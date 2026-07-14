import os
import re


def get_latest_json(dir_path, file_base):
    """
    Return the path to the latest <file_base>_XXXX.json file in dir_path.
    Raises FileNotFoundError if none found.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pattern = re.compile(rf"^{re.escape(file_base)}_(\d+)\.json$")

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
        raise FileNotFoundError(f"No '{file_base}_*.json' found in {dir_path}")

    return os.path.join(dir_path, latest_file)
