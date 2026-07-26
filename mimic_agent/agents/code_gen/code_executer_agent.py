import os
import re

import subprocess

from langchain_core.messages import get_buffer_string, AIMessage
from agents.code_gen.state import AgentState
from utils.tools import extract_python_code_from_message

from typing import Dict, Any


def make_code_executer_node(cfg):
    def CodeExecuterAgent(state: AgentState)->AgentState:
        """
        This agent is responsible for:
            1. Generating {skill}.py file in the Data Generation codebase
            2. Running the simulator
            3. Determining success: Store Yes if no bugs, else Store No and the execution error in the state
        :param state:
        :return:
        """
        try:
            # coder_output = json.loads(output)
            output = extract_python_code_from_message(get_buffer_string([state['messages'][-1]]))
            skill_class = state["skill_name"]

        except Exception as e:
            ai_message = AIMessage([f"Error parsing JSON output.\n Error: {e} \n Fix the error and try again."])
            return {
                'messages': [ai_message],
                'skill_name': state['skill_name'],
                'skill_version': state['skill_version'],
                'success': False,
                'num_critic_iters': state['num_critic_iters'],
                'rewrite_mode': state['rewrite_mode'],
                'skill_desc': state['skill_desc'],

            }

        message = write_to_file(state, output, cfg)

        if not message['success']:
            return {
                'messages': [message['messages']],
                'skill_name': state['skill_name'],
                'skill_version': state['skill_version'],
                'success': False,
                'num_critic_iters': state['num_critic_iters'],
                'rewrite_mode': state['rewrite_mode'],
                'skill_desc': state['skill_desc'],

            }

        result = run_skill(skill_class, skill_folder_version=state['skill_version'], cfg=cfg)

        if result['success']:
            return {
                'messages': [AIMessage("Success")],
                'skill_name': state['skill_name'],
                'skill_version': state['skill_version'],
                'success': True,
                'num_critic_iters': state['num_critic_iters'],
                'rewrite_mode': state['rewrite_mode'],
                'skill_desc': state['skill_desc'],

            }

        else:
            return {
                'messages': [AIMessage('Failure\n' + result['stderr'] + '\n' + result['exception'])],
                'skill_name': state['skill_name'],
                'skill_version': state['skill_version'],
                'success': False,
                'num_critic_iters': state['num_critic_iters'],
                'rewrite_mode': state['rewrite_mode'],
                'skill_desc': state['skill_desc'],

            }

    return CodeExecuterAgent


def write_to_file(state: AgentState, code: str, cfg: Dict[str, Any]):
    """
    Write generated skill code using versioned filenames:
    <skill_name>_0001.py, <skill_name>_0002.py, ...

    Returns success + error message if any.
    """

    ai_message = []

    skill_name = state["skill_name"]
    gait_dir = cfg["paths"]["gait_dir"]
    skill_dir = os.path.join(gait_dir, skill_name)

    os.makedirs(skill_dir, exist_ok=True)

    # --------------------------------------------------
    # Find existing versions
    # --------------------------------------------------
    pattern = re.compile(rf"{skill_name}_(\d+)\.py$")
    versions = []

    for fname in os.listdir(skill_dir):
        m = pattern.match(fname)
        if m:
            versions.append(int(m.group(1)))

    next_version = max(versions) + 1 if versions else 1
    fname = f"{skill_name}_{next_version:04d}.py"
    file_path = os.path.join(skill_dir, fname)

    # --------------------------------------------------
    # Write file
    # --------------------------------------------------
    try:
        with open(file_path, "w") as f:
            f.write(code)

        success = True

    except Exception as e:
        ai_message = [
            AIMessage(
                content=(
                    "Failed to write python code to file.\n"
                    f"Error: {e}\n"
                    "Fix the error and try again."
                )
            )
        ]
        success = False

    return {
        "messages": ai_message,
        "success": success,
        "file_path": file_path if success else None,
        "version": next_version if success else None,
    }


def run_skill(skill_name: str, skill_folder_version: str, cfg):
    python_bin = cfg["simulator"]["python_bin"]

    try:
        result = subprocess.run(
            [
                python_bin,
                "run.py",
                "--mode", skill_name,
                "--record", "True",
                "--version", skill_folder_version,
            ],
            cwd=cfg["paths"]["simulator_dir"],
            check=True,
            capture_output=True,
            text=True,
        )

        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "returncode": e.returncode,
            "exception": str(e),
        }

