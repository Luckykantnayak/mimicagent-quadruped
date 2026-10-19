import ast
import os
import re

import subprocess

from langchain_core.messages import get_buffer_string, AIMessage
from agents.code_gen.state import AgentState
from utils.tools import extract_python_code_from_message
from agents.code_gen.repair_context import response_text

from typing import Dict, Any


def validate_generated_code(code: str):
    """
    Reject output that cannot possibly run, BEFORE writing it to disk and paying
    for a simulator round-trip. Returns an actionable message for the critic, or
    None if the code is worth executing.

    Without this, a truncated / prose-only / empty response burns a whole repair
    iteration and surfaces as a misleading traceback from deep inside the sim.
    """
    if not code or not code.strip():
        return (
            "Your response contained no Python code at all. "
            "Reply with ONLY a single complete Python file inside one "
            "```python ... ``` fenced block, and nothing else."
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (
            f"The generated file is not valid Python: SyntaxError: {e.msg} "
            f"(line {e.lineno}). The file is most likely truncated or contains "
            f"prose outside the code block. Reply with ONLY a single COMPLETE "
            f"Python file inside one ```python ... ``` fenced block."
        )

    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if not any(n.endswith("MotionGenerator") for n in classes):
        return (
            "The generated file defines no class whose name ends in "
            f"'MotionGenerator' (found: {classes or 'none'}). Define exactly one, "
            "declared as: class <SKILL_NAME>_MotionGenerator(BaseMotionGenerator)."
        )

    imports_base = any(
        isinstance(n, ast.ImportFrom) and n.module == "motion.base"
        and any(a.name == "BaseMotionGenerator" and a.asname in (None, "BaseMotionGenerator")
                for a in n.names) for n in tree.body)
    candidates = [n for n in tree.body if isinstance(n, ast.ClassDef)
                  and n.name.endswith("MotionGenerator")]
    if not imports_base or len(candidates) != 1 or candidates[0].name == "BaseMotionGenerator":
        return "Import BaseMotionGenerator from motion.base and define exactly one concrete *MotionGenerator subclass; no substitute base classes."
    cls = candidates[0]
    if not any(isinstance(b, ast.Name) and b.id == "BaseMotionGenerator" for b in cls.bases):
        return "The motion class must directly subclass the imported BaseMotionGenerator."
    if "__abstractmethods__" in code or "update_abstractmethods" in code:
        return "Do not modify abstract methods dynamically; implement the required callback explicitly."
    methods = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    forbidden = methods & {"step", "reset", "compute_all_foot_positions_body_frame", "compute_all_foot_positions_world_frame"}
    if forbidden:
        return f"Inherit the fixed pipeline methods unchanged: {sorted(forbidden)}. Override only the per-foot callback, phase, or update_base_motion."
    if "compute_foot_position_body_frame" not in methods:
        return "Implement compute_foot_position_body_frame(self, leg_name, phase) explicitly, returning a finite (3,) body-frame vector for every leg."
    return None


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
            output = extract_python_code_from_message(response_text(state['messages'][-1]))
            skill_class = state["skill_name"]

        except Exception as e:
            ai_message = AIMessage([f"Error parsing JSON output.\n Error: {e} \n Fix the error and try again."])
            return {
                'messages': [ai_message],
                'skill_name': state['skill_name'],
                'skill_version': state['skill_version'],
                'success': False,
                # count the failed attempt so the crash-repair loop terminates
                'num_critic_iters': (state['num_critic_iters'] or 0) + 1,
                'rewrite_mode': state['rewrite_mode'],
                'skill_desc': state['skill_desc'],

            }

        # Guard: don't write or simulate output that can't run. Hand the actual
        # defect to the critic instead of a downstream traceback.
        problem = validate_generated_code(output)
        if problem:
            print(f"[code_executer] rejected generated code: {problem.splitlines()[0]}")
            return {
                'messages': [AIMessage("Failure\n" + problem)],
                'skill_name': state['skill_name'],
                'skill_version': state['skill_version'],
                'success': False,
                'num_critic_iters': (state['num_critic_iters'] or 0) + 1,
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
                # count the failed attempt so the crash-repair loop terminates
                'num_critic_iters': (state['num_critic_iters'] or 0) + 1,
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
                # count the failed attempt so the crash-repair loop terminates
                'num_critic_iters': (state['num_critic_iters'] or 0) + 1,
                'rewrite_mode': state['rewrite_mode'],
                'skill_desc': state['skill_desc'],

            }

    def node(state):
        result = CodeExecuterAgent(state)
        result["current_skill_code"] = extract_python_code_from_message(response_text(state["messages"][-1]))
        result["execution_feedback"] = response_text(result["messages"][0])
        # A new attempt invalidates the preceding attempt's score and diagnosis.
        result["evaluation_feedback"] = ""
        result["diagnosis"] = ""
        return result

    return node


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
                "simulator/run.py",
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

