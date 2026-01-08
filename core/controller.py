"""
gait_controller_runner.py

Main runner that:
1. Receives velocity commands
2. Uses MotionGenerator to compute root pose and foot positions
3. Solves IK to get joint angles
4. Updates simulation
"""
import mujoco
import numpy as np
import sys
from pathlib import Path

from utils.math_utils import quat_to_rotation_matrix
from gaits import get_motion_controller

# -------------------------------------------------
# Add project root to PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MotionControllerRunner:
    """
    Main controller that orchestrates:
    - Motion generation (root pose + foot positions)
    - IK solving
    - Simulation updates
    """

    def __init__(self, sim, args):
        self.sim = sim
        self.args = args
        self.model = sim.model
        self.data = sim.data

        # -------------------------------------------------
        # Foot & ground geom IDs
        # -------------------------------------------------

        # -------------------------------------------------
        # Leg names used by IK
        # -------------------------------------------------
        self.leg_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

        self.wheel_radius = 0.1040071605

        # -------------------------------------------------
        # IK Solver
        # -------------------------------------------------
        from utils.kinematics import MultiLinkGradientDescentIK

        self.ik = MultiLinkGradientDescentIK(
            self.model,
            self.data,
            self.leg_names,
        )

        # -------------------------------------------------
        # Initial foot positions (BODY FRAME)
        # -------------------------------------------------
        root_pos = self.data.qpos[0:3].copy()
        root_quat = self.data.qpos[3:7].copy()

        foot_positions_world = self.ik.get_foot_positions()
        # foot_positions_world[:, 2] += self.wheel_radius

        R = quat_to_rotation_matrix(root_quat)

        initial_feet_body = {}
        for i, leg_name in enumerate(self.leg_names):
            pos_world = foot_positions_world[i]
            pos_body = R.T @ (pos_world - root_pos)
            initial_feet_body[leg_name] = pos_body.copy()

        # -------------------------------------------------
        # Motion Generator
        # -------------------------------------------------
        Motion = get_motion_controller(args.mode, gaits_dir="gaits")

        self.motion_gen = Motion(
            initial_foot_positions_body=initial_feet_body,
            leg_names=self.leg_names,
        )

        self.motion_gen.reset(root_pos, root_quat)

        self.motion_gen.set_velocity_command(
            vx=getattr(args, "vx", 0.0),
            vy=getattr(args, "vy", 0.0),
            vz=getattr(args, "vz", 0.0),
        )

        self.motion_gen.set_angular_velocity_command(
            roll_rate=getattr(args, "roll_rate", 0.0),
            pitch_rate=getattr(args, "pitch_rate", 0.0),
            yaw_rate=getattr(args, "yaw_rate", 0.0),
        )

        # -------------------------------------------------
        # Timing
        # -------------------------------------------------
        self.dt = 1.0 / args.sim_freq
        self.frame = 0

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

        root_pos = motion_state["root_pos"]
        root_quat = motion_state["root_quat"]
        foot_positions_world = motion_state["foot_positions_world"]

        # Foot targets for IK
        foot_targets = np.zeros((4, 3))
        for i, leg_name in enumerate(self.leg_names):
            foot_targets[i] = foot_positions_world[leg_name]


        self.ik.calculate(foot_targets, debug=(self.frame % 60 == 0))

        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat

        self.frame += 1

        # print(foot_targets.shape)
        print("Foot tars", foot_targets[:, 2])
        print("Joints", len(self.data.qpos[7:]))

        return {
            "time": motion_state["time"],
            "root_pos": root_pos.copy(),
            "root_quat": root_quat.copy(),
            "joints": self.data.qpos[7:].copy(),
            "foot_positions_world": foot_targets.copy(),
        }

    # -------------------------------------------------
    # Contact detection
    # -------------------------------------------------
