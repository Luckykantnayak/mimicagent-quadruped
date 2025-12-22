"""
gait_controller_runner.py

Main runner that:
1. Receives velocity commands
2. Uses WalkingMotionGenerator to compute root pose and foot positions
3. Solves IK to get joint angles
4. Updates simulation
"""

import numpy as np
import sys
from pathlib import Path
from utils.math_utils import *
from gaits import get_motion_controller

# Add project root to PYTHONPATH
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
        """
        Args:
            sim: MuJoCo simulation object
            args: Configuration arguments with attributes:
                - mode: gait type (not used here, but kept for compatibility)
                - sim_freq: simulation frequency (Hz)
                - gait_freq: gait frequency (Hz)
                - step_length: step length (m)
                - step_height: step height (m)
                - vx, vy, vz: linear velocities (m/s) [optional]
                - roll_rate, pitch_rate, yaw_rate: angular rates (rad/s) [optional]
        """
        self.sim = sim
        self.args = args
        self.model = sim.model
        self.data = sim.data

        # -----------------------------
        # Leg names
        # -----------------------------
        self.leg_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

        # -----------------------------
        # IK Solver
        # -----------------------------
        from utils.kinematics import MultiLinkGradientDescentIK
        
        self.ik = MultiLinkGradientDescentIK(
            self.model,
            self.data,
            self.leg_names,
        )

        # -----------------------------
        # Get initial foot positions in body frame
        # -----------------------------
        root_pos = self.data.qpos[0:3].copy()
        root_quat = self.data.qpos[3:7].copy()
        
        # Get foot positions in world frame from current pose
        foot_positions_world = self.ik.get_foot_positions()
        
        # Convert to body frame
        R_mat = quat_to_rotation_matrix(root_quat)
        initial_feet_body = {}
        
        for i, leg_name in enumerate(self.leg_names):
            pos_world = foot_positions_world[i]
            pos_body = R_mat.T @ (pos_world - root_pos)
            initial_feet_body[leg_name] = pos_body.copy()

        # -----------------------------
        # Motion Generator
        # -----------------------------
        Motion = get_motion_controller(args.mode)
        self.motion_gen = Motion(
            initial_foot_positions_body=initial_feet_body,
            leg_names=self.leg_names,
            freq=args.gait_freq,
            duty_ratio=getattr(args, 'duty_ratio', 0.75),
            step_length=args.step_length,
            step_height=args.step_height,
            style=getattr(args, "skating_style", None)
        )
        
        
        # Initialize motion generator with current pose
        self.motion_gen.reset(root_pos, root_quat)
        
        # Set initial velocity commands
        self.motion_gen.set_velocity_command(
            vx=getattr(args, 'vx', 0.0),
            vy=getattr(args, 'vy', 0.0),
            vz=getattr(args, 'vz', 0.0)
        )
        
        self.motion_gen.set_angular_velocity_command(
            roll_rate=getattr(args, 'roll_rate', 0.0),
            pitch_rate=getattr(args, 'pitch_rate', 0.0),
            yaw_rate=getattr(args, 'yaw_rate', 0.0)
        )

        # -----------------------------
        # Timing
        # -----------------------------
        self.dt = 1.0 / args.sim_freq
        self.frame = 0

    def set_velocity_command(self, vx=0.0, vy=0.0, vz=0.0):
        """
        Set linear velocity command in world frame.
        
        Args:
            vx: forward velocity (m/s)
            vy: sideways velocity (m/s)
            vz: vertical velocity (m/s)
        """
        self.motion_gen.set_velocity_command(vx, vy, vz)

    def set_angular_velocity_command(self, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0):
        """
        Set angular velocity command in world frame.
        
        Args:
            roll_rate: roll angular velocity (rad/s)
            pitch_rate: pitch angular velocity (rad/s)
            yaw_rate: yaw angular velocity (rad/s)
        """
        self.motion_gen.set_angular_velocity_command(roll_rate, pitch_rate, yaw_rate)

    def reset(self):
        """Reset the controller to initial state."""
        root_pos = self.data.qpos[0:3].copy()
        root_quat = self.data.qpos[3:7].copy()
        self.motion_gen.reset(root_pos, root_quat)
        self.frame = 0

    def step(self):
        """
        Advance simulation by one control step.

        Returns:
            dict with state information for recording:
                - time: simulation time
                - root_pos: root position [x,y,z]
                - root_quat: root quaternion [w,x,y,z]
                - joints: joint angles
                - vel_world: commanded linear velocity
                - omega_world: commanded angular velocity
        """

        # ---------------------------------
        # Generate motion (root pose + foot positions)
        # ---------------------------------
        motion_state = self.motion_gen.step(self.dt)
        
        root_pos = motion_state['root_pos']
        root_quat = motion_state['root_quat']
        foot_positions_world = motion_state['foot_positions_world']

        # ---------------------------------
        # Convert foot dict to array for IK
        # ---------------------------------
        foot_targets_world = np.zeros((4, 3))
        for i, leg_name in enumerate(self.leg_names):
            foot_targets_world[i] = foot_positions_world[leg_name]

        # ---------------------------------
        # Solve IK
        # ---------------------------------
        self.ik.calculate(foot_targets_world, debug=(self.frame % 60 == 0))

        # ---------------------------------
        # Update simulation state
        # ---------------------------------
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat

        # ---------------------------------
        # Advance frame counter
        # ---------------------------------
        self.frame += 1

        # ---------------------------------
        # Return state for recording
        # ---------------------------------
        return {
            "time": motion_state['time'],
            "root_pos": root_pos.copy(),
            "root_quat": root_quat.copy(),
            "joints": self.data.qpos[7:].copy()
        }