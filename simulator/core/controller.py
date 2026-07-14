"""
core/controller.py

Main runner that:
1. Loads the robot spec (end-effectors, joint names, scene path)
2. Uses MotionGenerator to compute root pose and end-effector positions
3. Solves IK to get joint angles
4. Updates simulation
"""
import mujoco
import numpy as np
import os
import sys
from pathlib import Path

from utils.robot_config import load_robot_spec
from motion import get_motion_controller, get_example_controller


# -------------------------------------------------
# Add project root to PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MotionControllerRunner:
    """
    Main controller that orchestrates:
    - Motion generation (root pose + end-effector positions)
    - IK solving
    - Simulation updates
    """

    def __init__(self, sim, args):
        self.sim = sim
        self.args = args
        self.model = sim.model
        self.data = sim.data

        # -------------------------------------------------
        # Load robot spec
        # -------------------------------------------------
        robots_dir = getattr(args, "robots_dir", "robots")
        self.spec = load_robot_spec(args.robot, robots_dir=robots_dir)

        print(f"[Controller] Robot: {self.spec.name}")
        print(f"[Controller] End-effectors: {self.spec.ee_names}")

        # -------------------------------------------------
        # Validate: spec joint order must match model qpos[7:] order
        # (the recorder and quality checks depend on this invariant)
        # -------------------------------------------------
        self._check_joint_order()

        # -------------------------------------------------
        # IK Solver
        # -------------------------------------------------
        from utils.kinematics import MultiLinkGradientDescentIK

        self.ik = MultiLinkGradientDescentIK(
            self.model,
            self.data,
            self.spec.ee_bodies,
            sites=self.spec.ee_sites,
            joint_names=self.spec.actuated_joint_names,
        )

        # -------------------------------------------------
        # Initial end-effector positions in BODY FRAME
        # -------------------------------------------------
        base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.spec.base_body
        )
        if base_body_id < 0:
            raise ValueError(f"Base body '{self.spec.base_body}' not found in model")

        R_wb = self.data.xmat[base_body_id].reshape(3, 3)
        p_wb = self.data.xpos[base_body_id].copy()

        ee_positions_world = self.ik.get_ee_positions()

        initial_ee_body = {}
        for ee_name, pos_world in zip(self.spec.ee_names, ee_positions_world):
            pos_body = R_wb.T @ (pos_world - p_wb)
            initial_ee_body[ee_name] = pos_body.copy()

        print(f"[Controller] Initial EE positions (body frame):")
        for k, v in initial_ee_body.items():
            print(f"  {k}: {np.round(v, 4)}")

        # -------------------------------------------------
        # Motion Generator
        #   mode "example"        -> the robot's reference skill
        #   agent-generated skill -> skill_output/<robot>/<mode>/<mode>_<N>.py
        # -------------------------------------------------
        skill_dir = os.path.join("skill_output", self.spec.name, args.mode)
        if args.mode == "example" or not os.path.isdir(skill_dir):
            print(f"[Controller] Loading reference skill: {self.spec.example_skill}")
            Motion = get_example_controller(self.spec.example_skill)
        else:
            Motion = get_motion_controller(
                args.mode, gaits_dir=os.path.join("skill_output", self.spec.name)
            )

        self.motion_gen = Motion(init_ee_pos=initial_ee_body)

        # Each skill is self-contained: it owns its base motion (velocity /
        # yaw) inside update_base_motion. The controller does not inject any
        # external velocity command, so nothing overrides the skill.
        root_pos = self.data.qpos[0:3].copy()
        root_quat = self.data.qpos[3:7].copy()
        self.motion_gen.reset(root_pos, root_quat)

        # -------------------------------------------------
        # Timing
        # -------------------------------------------------
        self.dt = 1.0 / args.sim_freq
        self.frame = 0

    # -------------------------------------------------
    # Spec / model consistency
    # -------------------------------------------------
    def _check_joint_order(self):
        """Assert spec.joints matches the model's qpos[7:] joint order."""
        model_joints = []
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            model_joints.append(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            )
        if model_joints != self.spec.joint_names:
            raise ValueError(
                f"Robot spec joint order does not match model qpos order.\n"
                f"  model: {model_joints}\n"
                f"  spec : {self.spec.joint_names}"
            )

    # -------------------------------------------------
    # Commands
    # -------------------------------------------------
    def set_velocity_command(self, vx=0.0, vy=0.0, vz=0.0):
        self.motion_gen.set_velocity_command(vx, vy, vz)

    def set_angular_velocity_command(self, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0):
        self.motion_gen.set_angular_velocity_command(roll_rate, pitch_rate, yaw_rate)

    def reset(self):
        root_pos = self.data.qpos[0:3].copy()
        root_quat = self.data.qpos[3:7].copy()
        self.motion_gen.reset(root_pos, root_quat)
        self.frame = 0

    # -------------------------------------------------
    # Main step
    # -------------------------------------------------
    def step(self):
        motion_state = self.motion_gen.step(self.dt)

        root_pos  = motion_state["root_pos"]
        root_quat = motion_state["root_quat"]
        ee_positions_world = motion_state["ee_positions_world"]

        # Build IK target array in spec end-effector order
        ee_targets = np.zeros((len(self.spec.ee_names), 3))
        for i, ee_name in enumerate(self.spec.ee_names):
            ee_targets[i] = ee_positions_world[ee_name]

        self.ik.calculate(ee_targets, debug=(self.frame % 60 == 0))

        # Apply joint overrides from gait (skill-specific)
        joint_overrides = motion_state.get("joint_overrides", {})
        for joint_name, angle in joint_overrides.items():
            if joint_name in self.spec.joint_names:
                idx = self.spec.joint_names.index(joint_name)
                self.data.qpos[7 + idx] = angle

        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat

        self.frame += 1

        return {
            "time": motion_state["time"],
            "root_pos": root_pos.copy(),
            "root_quat": root_quat.copy(),
            "joints": self.data.qpos[7:].copy(),
            "ee_positions_world": ee_targets.copy(),
        }
