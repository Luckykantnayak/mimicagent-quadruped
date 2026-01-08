from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_JUMP_TWIST_SPIN_MotionGenerator(BaseMotionGenerator):
    """
    Continuous vertical jump with simultaneous pitch, roll, and yaw rotation.
    Legs push off together, tuck during flight, and land in a staggered stance.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        """
        Parameters
        ----------
        initial_foot_positions_body : dict
            Initial foot positions in BODY frame for each leg.
        """
        # Call base constructor to set up common state
        super().__init__(initial_foot_positions_body, freq=0.25)

        # Motion parameters
        self.takeoff_height = 2.          # meters to lift during takeoff
        self.flight_duration = 0.6         # fraction of cycle (0.2-0.8)
        self.landing_duration = 0.2        # fraction of cycle (0.8-1.0)
        self.total_cycle = 1.0

        # Rotation rates (rad/s) during flight
        self.pitch_rate = np.deg2rad(90.0)   # 90 deg/s
        self.roll_rate = np.deg2rad(90.0)    # 90 deg/s
        self.yaw_rate = np.deg2rad(90.0)     # 90 deg/s

        # Leg extension parameters
        self.landing_extension = 0.15       # meters to extend during landing

        # Staggered landing order
        self.landing_order = ["FL", "FR", "RL", "RR"]

    def update_base_motion(self, phase, dt):
        """
        Update base pose: vertical lift during takeoff,
        continuous rotation during flight, and deceleration during landing.
        """
        # Compute time within current cycle
        t_cycle = self.t

        # Vertical velocity profile
        if phase < 0.2:  # takeoff
            v_z = (self.takeoff_height / 0.2) * dt
        elif phase < 0.8:  # flight (no vertical motion)
            v_z = 0.0
        else:  # landing
            v_z = -(self.takeoff_height / 0.2) * dt

        self.vel_world = np.array([0.0, 0.0, v_z])

        # Rotation during flight
        if phase < 0.2:
            omega = np.array([0.0, 0.0, 0.0])
        elif phase < 0.8:
            omega = np.array([self.roll_rate, self.pitch_rate, self.yaw_rate])
        else:
            omega = np.array([0.0, 0.0, 0.0])

        self.omega_world = omega

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
        Compute foot target in BODY frame for each leg based on sub-phase.
        """
        # Base foot position (initial stance)
        base_pos = self.base_init_feet_pos[leg_name].copy()

        # Takeoff: push off together, maintain stance
        if phase < 0.2:
            # No horizontal movement; foot stays at base position
            return base_pos

        # Flight: tuck legs close to body (reduce x and y)
        elif phase < 0.8:
            # Tuck factor decreases from 1 to 0 over flight
            tuck_factor = 1.0 - (phase - 0.2) / self.flight_duration
            # Reduce horizontal offset proportionally
            pos = base_pos.copy()
            pos[0] *= tuck_factor  # x (forward)
            pos[1] *= tuck_factor  # y (lateral)
            return pos

        # Landing: extend sequentially
        else:
            # Determine which leg is extending based on landing order and phase
            landing_phase = (phase - 0.8) / self.landing_duration
            leg_index = next(i for i, name in enumerate(self.landing_order) if leg_name.startswith(name))
            # Each leg starts extending when landing_phase >= (index / 4)
            start = leg_index * 0.25
            end = start + 0.25

            if landing_phase < start:
                # Leg still in stance
                return base_pos
            elif landing_phase <= end:
                # Extend linearly during this quarter
                progress = (landing_phase - start) / 0.25
                pos = base_pos.copy()
                pos[2] += self.landing_extension * progress  # lift foot upward
                return pos
            else:
                # Leg fully extended (touching ground)
                pos = base_pos.copy()
                pos[2] += self.landing_extension
                return pos

    def compute_phase(self, t):
        """
        Override to keep phase in [0,1] over continuous time.
        """
        return (self.freq * t) % 1.0
