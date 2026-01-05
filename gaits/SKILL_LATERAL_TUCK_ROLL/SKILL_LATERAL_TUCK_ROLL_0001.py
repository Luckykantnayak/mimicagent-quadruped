from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_LATERAL_TUCK_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Lateral tuck roll motion generator.

    The robot performs a continuous sideways rolling motion, rotating about its
    longitudinal axis while translating laterally to the left. All four legs
    remain tucked close to the body throughout the roll to minimize rotation
    radius and enable fluid rotation through inverted positions.

    Phase breakdown:
      [0.00, 0.25]: Right side lift initiation
      [0.25, 0.50]: Inverted position
      [0.50, 0.75]: Left side lift continuation
      [0.75, 1.00]: Upright return preparation
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        # Initialize base class
        base_init_feet_pos = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        super().__init__(base_init_feet_pos, freq=0.5)

        self.leg_names = leg_names

        # Store base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Roll motion parameters
        self.roll_rate = -2.0 * np.pi  # -360 deg/s for one full rotation per cycle
        self.lateral_velocity = -0.3  # Leftward drift (negative y in world frame)

        # Leg tuck parameters
        self.tuck_radius_lateral = 0.08  # How close to body centerline (y=0 in body frame)
        self.tuck_height = -0.05  # How close to body core (z in body frame, negative is up toward body)
        self.extend_factor = 0.6  # Partial extension at cycle end (0=fully tucked, 1=fully extended)

    def update_base_motion(self, phase, dt):
        """
        Update base motion using continuous roll rate and lateral translation.

        Roll rate is constant throughout to maintain momentum.
        Lateral velocity provides leftward drift.
        """
        # Constant lateral velocity to the left (negative y in world frame)
        vx = 0.0
        vy = self.lateral_velocity
        vz = 0.0

        # Add slight downward velocity in final phase to re-establish ground proximity
        if phase >= 0.75:
            vz = -0.2 * ((phase - 0.75) / 0.25)

        # Constant negative roll rate (counterclockwise from robot's perspective)
        roll_rate = self.roll_rate
        pitch_rate = 0.0
        yaw_rate = 0.0

        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])

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
        Compute foot position in body frame for given leg and phase.

        Motion strategy:
          - Phase [0.0, 0.25]: Rapid tuck from base position
          - Phase [0.25, 0.75]: Maintain maximal tuck
          - Phase [0.75, 1.0]: Partial extension to prepare for next cycle

        Tuck motion brings feet inward toward body centerline (y→0) and
        upward toward body core (z→negative, closer to body).
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()

        # Determine tuck amount based on phase
        if phase < 0.25:
            # Rapid tuck during initiation
            tuck_progress = phase / 0.25
            tuck_amount = self._smooth_step(tuck_progress)
        elif phase < 0.75:
            # Maintain full tuck through inverted and continuation
            tuck_amount = 1.0
        else:
            # Partial extension during upright return
            extend_progress = (phase - 0.75) / 0.25
            tuck_amount = 1.0 - self.extend_factor * self._smooth_step(extend_progress)

        # Compute tucked position
        foot_pos = base_pos.copy()

        # Tuck lateral (y): bring toward centerline
        lateral_offset = base_pos[1]
        if abs(lateral_offset) > self.tuck_radius_lateral:
            target_y = np.sign(lateral_offset) * self.tuck_radius_lateral
            foot_pos[1] = base_pos[1] + tuck_amount * (target_y - base_pos[1])

        # Tuck vertical (z): bring upward toward body core
        vertical_offset = base_pos[2]
        target_z = self.tuck_height
        foot_pos[2] = base_pos[2] + tuck_amount * (target_z - base_pos[2])

        # Tuck longitudinal (x): bring slightly toward center
        longitudinal_offset = base_pos[0]
        target_x = longitudinal_offset * 0.7  # Reduce forward/back extension
        foot_pos[0] = base_pos[0] + tuck_amount * (target_x - base_pos[0])

        return foot_pos

    def _smooth_step(self, t):
        """
        Smooth step function for smooth transitions.
        Uses smoothstep interpolation: 3t^2 - 2t^3
        """
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)