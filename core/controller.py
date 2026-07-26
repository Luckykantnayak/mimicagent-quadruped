"""
core/controller.py

Main runner that:
1. Loads robot config (leg names, joint names, scene path)
2. Normalizes leg names to canonical FL/FR/RL/RR for all gait files
3. Uses MotionGenerator to compute root pose and foot positions
4. Denormalizes back to XML names before sending to IK / MuJoCo
5. Solves IK to get joint angles
6. Updates simulation
"""
import mujoco
import numpy as np
import sys
from pathlib import Path

from utils.math_utils import quat_to_rotation_matrix
from utils.robot_config import load_robot_config, normalize_leg_name, denormalize_leg_name
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
    - Leg-name normalization (robot XML names ↔ canonical FL/FR/RL/RR)
    - IK solving
    - Simulation updates
    """

    def __init__(self, sim, args):
        self.sim = sim
        self.args = args
        self.model = sim.model
        self.data = sim.data

        # -------------------------------------------------
        # Load robot config
        # -------------------------------------------------
        robots_dir = getattr(args, "robots_dir", "robots")
        self.robot_cfg = load_robot_config(args.robot, robots_dir=robots_dir)

        # XML body names (as they appear in the MuJoCo model)
        self.xml_leg_names = (
            self.robot_cfg.calf_bodies +
            self.robot_cfg.hand_bodies
        )

        # Canonical names used by ALL gait files (FL/FR/RL/RR)
        self.canonical_leg_names = [normalize_leg_name(n) for n in self.xml_leg_names]

        # Mapping: canonical → xml  (used when sending targets back to IK)
        self.canonical_to_xml = {
            normalize_leg_name(xml): xml for xml in self.xml_leg_names
        }
        # Mapping: xml → canonical  (used when reading positions from sim)
        self.xml_to_canonical = {v: k for k, v in self.canonical_to_xml.items()}

        print(f"[Controller] Robot: {self.robot_cfg.name}")
        print(f"[Controller] XML leg names:       {self.xml_leg_names}")
        print(f"[Controller] Canonical leg names: {self.canonical_leg_names}")

        self.wheel_radius = self.robot_cfg.wheel_radius

        # -------------------------------------------------
        # IK Solver  (uses XML body/site names)
        # -------------------------------------------------
        from utils.kinematics import MultiLinkGradientDescentIK
        all_sites = self.robot_cfg.foot_sites + self.robot_cfg.hand_sites

        self.ik = MultiLinkGradientDescentIK(
            self.model,
            self.data,
            self.xml_leg_names,
            foot_sites=all_sites,
            joint_names=self.robot_cfg.joint_names,
        )

        # -------------------------------------------------
        # Initial foot/hand positions in BODY FRAME
        # -------------------------------------------------
        base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.robot_cfg.base_body
        )
        if base_body_id < 0:
            raise ValueError(f"Base body '{self.robot_cfg.base_body}' not found in model")

        R_wb = self.data.xmat[base_body_id].reshape(3, 3)
        p_wb = self.data.xpos[base_body_id].copy()

        foot_positions_world = self.ik.get_foot_positions()

        initial_feet_body = {}
        for xml_name, pos_world in zip(self.xml_leg_names, foot_positions_world):
            canonical = self.xml_to_canonical[xml_name]
            pos_body = R_wb.T @ (pos_world - p_wb)
            initial_feet_body[canonical] = pos_body.copy()

        print(f"[Controller] Initial EE positions (body frame):")
        for k, v in initial_feet_body.items():
            print(f"  {k}: {np.round(v, 4)}")

        # -------------------------------------------------
        # Motion Generator  (receives canonical leg names)
        # -------------------------------------------------
        Motion = get_motion_controller(args.mode, gaits_dir="gaits")

        self.motion_gen = Motion(
            initial_foot_positions_body=initial_feet_body,
            leg_names=self.canonical_leg_names,
        )

        root_pos = self.data.qpos[0:3].copy()
        root_quat = self.data.qpos[3:7].copy()
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

        root_pos  = motion_state["root_pos"]
        root_quat = motion_state["root_quat"]
        foot_positions_world = motion_state["foot_positions_world"]

        # Build IK target array in XML leg order
        foot_targets = np.zeros((len(self.xml_leg_names), 3))
        for i, xml_name in enumerate(self.xml_leg_names):
            canonical = self.xml_to_canonical[xml_name]
            foot_targets[i] = foot_positions_world[canonical]

        self.ik.calculate(foot_targets, debug=(self.frame % 60 == 0))

        # Apply joint overrides from gait (skill-specific)
        joint_overrides = motion_state.get("joint_overrides", {})
        for joint_name, angle in joint_overrides.items():
            if joint_name in self.robot_cfg.joint_names:
                idx = self.robot_cfg.joint_names.index(joint_name)
                self.data.qpos[7 + idx] = angle

        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat

        self.frame += 1

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