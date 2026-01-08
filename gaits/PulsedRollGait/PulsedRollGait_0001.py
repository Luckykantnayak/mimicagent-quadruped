from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_5_MotionGenerator(BaseMotionGenerator):
    """
    Continuous torsional slide with coordinated passive leg swings.
    Base twists left and right while legs perform forward/backward
    passive motions.  The motion is defined over a single phase cycle
    [0,1] with the following sub‑phases:

        0.0 – 0.3   : left twist, forward swing
        0.3 – 0.6   : right twist, backward swing
        0.6 – 0.9   : pause (legs stationary), twist continues to accumulate
        0.9 – 1.0   : return to neutral twist, legs remain passive
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        """
        Parameters
        ----------
        initial_foot_positions_body : dict
            Mapping from leg name to 3‑D numpy array of the foot position
            expressed in the body frame at rest.
        """
        # Call base constructor with initial foot positions and desired frequency
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Motion parameters
        self.step_length = 0.1   # forward/backward swing distance (m)
        self.step_height = 0.1   # vertical lift during swing (m)

        # Twist parameters
        self.yaw_amp = 5.        # maximum yaw rate (rad/s)
        self.torque_phase_start = 0.0
        self.left_twist_end   = 0.3
        self.right_twist_end  = 0.6
        self.pause_end        = 0.9

    def update_base_motion(self, phase, dt):
        """
        Update base pose using piecewise yaw rate.
        The base twists left during the first third, right during the
        second third, continues to accumulate twist during pause,
        and then returns to neutral over the final quarter.
        """
        # Determine yaw rate based on current phase
        if phase < self.left_twist_end:
            # Left twist: negative yaw rate, linearly increasing magnitude
            progress = phase / self.left_twist_end
            yaw_rate = -self.yaw_amp * (1.0 - progress)
        elif phase < self.right_twist_end:
            # Right twist: positive yaw rate, linearly increasing magnitude
            progress = (phase - self.left_twist_end) / (
                self.right_twist_end - self.left_twist_end
            )
            yaw_rate = self.yaw_amp * (1.0 - progress)
        elif phase < self.pause_end:
            # Pause: maintain current twist direction (zero yaw rate)
            yaw_rate = 0.0
        else:
            # Return to neutral: linearly decrease yaw rate towards zero
            progress = (phase - self.pause_end) / (
                1.0 - self.pause_end
            )
            yaw_rate = self.yaw_amp * (1.0 - progress)

        # No linear motion; only yaw
        self.vel_world = np.zeros(3)
        self.omega_world = np.array([0.0, 0.0, yaw_rate])

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
        Compute passive foot trajectory in body frame.
        All legs perform the same forward/backward swing pattern
        synchronized with the base twist sub‑phases.
        """
        foot = self.base_init_feet_pos[leg_name].copy()

        # Forward swing: 0.0 – 0.3
        if phase < self.left_twist_end:
            progress = phase / self.left_twist_end
            foot[0] += self.step_length * (progress - 0.5)
            # Lift the foot slightly
            angle = np.pi * progress
            foot[2] += self.step_height * np.sin(angle)

        # Backward swing: 0.3 – 0.6
        elif phase < self.right_twist_end:
            progress = (phase - self.left_twist_end) / (
                self.right_twist_end - self.left_twist_end
            )
            foot[0] -= self.step_length * (progress - 0.5)
            angle = np.pi * progress
            foot[2] += self.step_height * np.sin(angle)

        # Pause: 0.6 – 1.0 (no change)
        else:
            pass

        return foot
