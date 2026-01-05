from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class CorkscrewBoundMotionGenerator(BaseMotionGenerator):
    """
    Continuous phase-based kinematic plan for a diagonal bound with a corkscrew base twist.
    Implements the motion described in the planner specification.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        """
        Parameters
        ----------
        initial_foot_positions_body : dict
            Mapping from leg name to 3‑D position in the body frame.
        """
        # Call base constructor
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Use provided leg names (order matters)
        self.leg_names = leg_names

        # Group legs by index instead of hard-coded names
        # group1: leg 0 & 3, group2: leg 1 & 2
        self.phase_offsets = {
            self.leg_names[0]: 0.0,
            self.leg_names[3]: 0.0,
            self.leg_names[1]: 0.25,
            self.leg_names[2]: 0.25,
        }

        # Motion parameters
        self.step_length = 0.15   # forward step length during push_off
        self.step_height = 0.10   # lift height during flight
        self.lateral_offset = 0.05  # lateral scissor amplitude

        # Base motion parameters
        self.push_off_speed = 2.   # forward speed during push_off
        self.pitch_amplitude = np.deg2rad(15)  # pitch up during flight
        self.yaw_amplitude = np.deg2rad(20)    # yaw during flight

        # Time tracking for base velocity functions
        self.t = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base pose using simple kinematic primitives:
          * Forward translation during push_off (phase 0–0.2)
          * Pitch up and yaw rotation during flight (phase 0.2–0.4)
          * Deceleration to neutral during landing and reset
        """
        # Forward translation during push_off
        if phase < 0.20:
            vx = self.push_off_speed
        else:
            vx = 0.0

        # Pitch and yaw during flight
        if 0.20 <= phase < 0.40:
            # Map phase to [0,1] within flight interval
            p = (phase - 0.20) / 0.20
            pitch_rate = self.pitch_amplitude * np.sin(np.pi * p)
            yaw_rate = self.yaw_amplitude * np.sin(np.pi * p)
        else:
            pitch_rate = 0.0
            yaw_rate = 0.0

        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, pitch_rate, yaw_rate])

        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot target in body frame for the given leg and global phase.
        Implements push_off, flight, landing, and reset behaviors.
        """
        # Local leg phase with offset
        local_phase = (phase + self.phase_offsets[leg_name]) % 1.0

        # Base foot position (neutral)
        pos = self.base_init_feet_pos[leg_name].copy()

        # Push_off (0–0.2)
        if local_phase < 0.20:
            progress = local_phase / 0.20
            # Forward extension
            pos[0] -= self.step_length * (progress - 0.5)
            # Slight lift
            pos[2] += self.step_height * progress

        # Flight (0.20–0.40)
        elif 0.20 <= local_phase < 0.40:
            progress = (local_phase - 0.20) / 0.20
            # Lateral scissor motion
            lateral = self.lateral_offset * np.sin(np.pi * progress)
            if leg_name.startswith(("FL", "RR")):
                pos[1] += lateral
            else:
                pos[1] -= lateral
            # Maintain forward swing
            pos[0] += self.step_length * (progress - 0.5)
            # Lift to flight height
            pos[2] += self.step_height * np.sin(np.pi * progress)

        # Landing (0.40–0.60)
        elif 0.40 <= local_phase < 0.60:
            progress = (local_phase - 0.40) / 0.20
            # Forward swing to re‑contact
            pos[0] += self.step_length * (progress - 0.5)
            # Slight descent
            pos[2] -= self.step_height * progress

        # Reset (0.60–1.00)
        else:
            # Return to neutral stance
            pos = self.base_init_feet_pos[leg_name].copy()

        return pos

    def compute_all_foot_positions_body_frame(self, phase):
        """
        Override to use the custom leg motion logic.
        """
        return {
            leg: self.compute_foot_position_body_frame(leg, phase)
            for leg in self.leg_names
        }