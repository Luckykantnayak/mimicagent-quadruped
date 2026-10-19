import glob
import json
import os

from agents.code_gen.state import AgentState
from utils.test_motion_quality import evaluate_trajectory, save_report
from utils.skill_fidelity import append_fidelity
from utils.read_latest_version import get_latest_skill_file

from langchain_core.messages import AIMessage

import pickle as pkl

def make_unit_test_node(cfg):

    def run_unit_test(state: AgentState) -> AgentState:
        skill_version = state["skill_version"]
        skill_name = state["skill_name"]
        # The simulator names its output "<robot>_<mode>.pkl", so match on the suffix
        # rather than assuming the bare skill name.
        traj_dir = os.path.join(cfg["paths"]["trajectory_dir"], skill_version)
        matches = sorted(glob.glob(os.path.join(traj_dir, f"*{skill_name}.pkl")))
        if not matches:
            raise FileNotFoundError(
                f"No recorded trajectory for '{skill_name}' in {traj_dir}"
            )
        fn = matches[-1]

        # Load trajectory
        with open(fn, "rb") as f:
            traj = pkl.load(f)

        # Get latest motion planner JSON file
        base_dir = cfg["paths"]["motion_planner_output_dir"]
        motion_plan_path = get_latest_skill_file(base_dir, skill_version, skill_name)

        with open(motion_plan_path, "r") as f:
            motion_plan = json.load(f)

        # Evaluate trajectory with unit tests
        joint_limits = {
            "FL_hip": (-1.0472, 1.0472),
            "FL_thigh": (-1.5708, 3.4907),
            "FL_calf": (-2.7227, -0.83776),

            "FR_hip": (-1.0472, 1.0472),
            "FR_thigh": (-1.5708, 3.4907),
            "FR_calf": (-2.7227, -0.83776),

            "RL_hip": (-1.0472, 1.0472),
            "RL_thigh": (-0.5236, 4.5379),
            "RL_calf": (-2.7227, -0.83776),

            "RR_hip": (-1.0472, 1.0472),
            "RR_thigh": (-0.5236, 4.5379),
            "RR_calf": (-2.7227, -0.83776),
        }

        # Extract skill specific constraints - True if applies, else False
        try:
            # config_closure = motion_plan["motion_plan"]["constraints"]["cycle_closure"]["enabled"]
            feet_airtime = motion_plan["motion_plan"]["constraints"]["all_feet_off_ground_consistency"]["enabled"]

            # assert isinstance(config_closure, bool), f"config_closure: Expected bool, got {type(config_closure)}"
            assert isinstance(feet_airtime, bool), f"feet_airtime: Expected bool, got {type(feet_airtime)}"

        except:
            print("Using default configuration for feet_airtime and config_closure (False)")
            # config_closure = False
            feet_airtime = False

        constraints = {
            # "configuration_closure": config_closure,
            # "relative_foot_closure": config_closure,
            "all_feet_airtime": feet_airtime,
        }


        # Get rewrite mode
        feedback = evaluate_trajectory(traj=traj, joint_limits=joint_limits, skill_test_config=constraints, dt=1/200)
        feedback = append_fidelity(feedback, traj,
                                   motion_plan["motion_plan"].get("success_criteria"),
                                   cfg["paths"]["simulator_dir"])
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

        out_dir = cfg["paths"]["results_dir"]
        skill_out_dir = os.path.join(out_dir, skill_version, skill_name)
        os.makedirs(skill_out_dir, exist_ok=True)

        # save report
        save_report(feedback, skill_name=state["skill_name"], out_dir=skill_out_dir)

        return {
            "evaluation_feedback": json.dumps(critic_payload),
            "messages": [AIMessage(json.dumps(critic_payload))],
            'skill_name': state['skill_name'],
            'skill_version': state['skill_version'],
            'success': state['success'],
            "num_critic_iters": (1 if state['num_critic_iters'] is None else state['num_critic_iters'] + 1),
            'rewrite_mode': rewrite_mode,
            'skill_desc': state['skill_desc'],
        }

    return run_unit_test
