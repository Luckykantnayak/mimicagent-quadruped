from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_SLITHER_CRAWL_MotionGenerator(BaseMotionGenerator):
    """
    Continuous lateral body wave with minimal leg lift.
    Base follows a sinusoidal lateral displacement along the body axis (x).
    Legs maintain ground contact and provide small tangential pushes phase‑shifted
    to match the body wave crest.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        """
        Parameters
        ----------
        initial_foot_positions_body : dict
            Mapping leg name -> 3‑D position in BODY frame.
        """
        # Call base constructor
        super().__init__(initial_foot_positions_body, freq=1.0)

        # Gait parameters
        self.duty = 1.0                     # always on ground
        self.wave_amp = 0.05                # lateral displacement amplitude (m)
        self.wave_freq = 2.0                # frequency of body wave (Hz)
        self.base_speed = 0.5  # m/s (tune)
        self.base_height = 0.3
        self.height_amp = 0.03

        # Foot push amplitude along body axis
        self.push_amp = 0.05                # small tangential push (m)

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

        # Store base foot positions (BODY frame)
        self.base_feet_pos_body = {
            k: v.copy() for k, v in initial_foot_positions_body.items()
        }

    def update_base_motion(self, phase, dt):
        """
        Base follows a sinusoidal lateral motion along the body x‑axis.
        No vertical or angular movement.
        """
        # Lateral velocity in BODY frame
        # vx_body = self.wave_amp * 2 * np.pi * self.wave_freq * np.cos(2 * np.pi * self.wave_freq * self.t)
        vx_body = (
                self.base_speed
                + self.wave_amp * 2 * np.pi * self.wave_freq
                * np.cos(2 * np.pi * self.wave_freq * self.t)
        )

        # Convert to WORLD frame
        vel_body = np.array([vx_body, 0.0, 0.0])
        self.vel_world = body_to_world_velocity(vel_body, self.root_quat)

        # No angular velocity
        self.omega_world = np.zeros(3)

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
        Maintain ground contact and apply a small tangential push
        that is phase‑shifted to match the body wave crest.
        """
        # Base foot position
        foot = self.base_feet_pos_body[leg_name].copy()

        # Compute leg phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0

        # Tangential push along body x‑axis
        push = self.push_amp * np.sin(2 * np.pi * self.wave_freq * (self.t + leg_phase / self.wave_freq))
        foot[0] += push

        # No vertical lift; keep z unchanged
        return foot

    def compute_phase(self, t):
        """
        Override to use continuous phase in [0,1] based on wave frequency.
        """
        return (self.wave_freq * t) % 1.0
