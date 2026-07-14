import json
import os

from agents.code_gen.state import AgentState
from utils.test_motion_quality import evaluate_trajectory, save_report
from utils.read_latest_version import get_latest_json
from utils.robot_spec import load_robot_spec

from langchain_core.messages import AIMessage

import pickle as pkl

def make_unit_test_node(cfg):

    spec = load_robot_spec(cfg)

    def run_unit_test(state: AgentState) -> AgentState:
        skill_name = state["skill_name"]
        # All artifacts for a skill live in skill_output/<robot>/<skill>/
        skill_dir = os.path.join(
            cfg["paths"]["skill_output_dir"], cfg["robot"]["name"], skill_name
        )
        fn = os.path.join(skill_dir, f"{skill_name}.pkl")

        # Load trajectory
        with open(fn, "rb") as f:
            traj = pkl.load(f)

        # Get latest motion plan JSON
        motion_plan_path = get_latest_json(skill_dir, "motion_plan")

        with open(motion_plan_path, "r") as f:
            motion_plan = json.load(f)

        # Extract skill specific constraints - True if applies, else False
        try:
            feet_airtime = motion_plan["motion_plan"]["constraints"]["all_feet_off_ground_consistency"]["enabled"]

            assert isinstance(feet_airtime, bool), f"feet_airtime: Expected bool, got {type(feet_airtime)}"

        except:
            print("Using default configuration for feet_airtime (False)")
            feet_airtime = False

        constraints = {
            "all_ee_airtime": feet_airtime,
        }

        # Get rewrite mode
        dt = 1.0 / traj["fps"]
        feedback = evaluate_trajectory(traj=traj, spec=spec, dt=dt, skill_test_config=constraints)
        summary = feedback["summary"]

        if not summary["passed_all_discrete"]:
            rewrite_mode = "structural"
        elif not summary["passed_smoothness"]:
            rewrite_mode = "parametric"
        else:
            rewrite_mode = "none"


        # Extract failures
        discrete_failures = [
            t for t in feedback["tests"]
            if t["category"] == "discrete"
               and not t["passed"]
               and not t["skipped"]
        ]

        smooth_failures = [
            t for t in feedback["tests"]
            if t["category"] == "smoothness"
               and t["metrics"].get("mean_excess", 0.0) > 0
        ]


        # Form feedback
        critic_payload = dict(
            rewrite_mode=rewrite_mode,

            summary=dict(
                passed_all=summary["passed_all"],
                passed_all_discrete=summary["passed_all_discrete"],
                passed_smoothness=summary["passed_smoothness"],
                total_reward=summary["total_reward"],
            ),

            discrete_failures=[
                {
                    "name": t["name"],
                    "violations": t["violations"],
                    "metrics": t["metrics"],
                    "feedback": t["feedback"],
                }
                for t in discrete_failures
            ],

            smoothness_failures=[
                {
                    "name": t["name"],
                    "mean_excess": t["metrics"]["mean_excess"],
                    "details": t["metrics"],
                }
                for t in smooth_failures
            ],
        )

        os.makedirs(skill_dir, exist_ok=True)

        # save quality report as report_NNNN.json in the skill dir
        save_report(feedback, skill_name="report", out_dir=skill_dir)

        return {
            "messages": [AIMessage(json.dumps(critic_payload))],
            'skill_name': state['skill_name'],
            'skill_version': state['skill_version'],
            'success': state['success'],
            "num_critic_iters": (1 if state['num_critic_iters'] is None else state['num_critic_iters'] + 1),
            'rewrite_mode': rewrite_mode,
            'skill_desc': state['skill_desc'],
        }

    return run_unit_test
