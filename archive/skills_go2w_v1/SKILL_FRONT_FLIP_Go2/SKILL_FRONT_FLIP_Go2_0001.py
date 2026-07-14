from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_FRONT_FLIP_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Front flip skill: synchronized four-leg aerial rotation.
    
    Phase structure:
      0.0-0.2:  Crouch/load - all legs flex deeply
      0.2-0.35: Explosive takeoff - legs extend, base accelerates up/forward, pitch rate initiates
      0.35-0.7: Airborne tuck - legs retract, body rotates ~360° in pitch
      0.7-0.9:  Pre-landing extension - legs extend to reach ground
      0.9-1.0:  Landing absorption - all feet contact, legs flex to absorb impact
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Front flip cycle duration ~2 seconds

        # Store base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Crouch parameters
        self.crouch_depth = 0.10  # meters, vertical retraction during crouch
        self.crouch_forward_shift = 0.02  # front legs shift slightly forward

        # Tuck parameters
        self.tuck_retraction_x = 0.08  # horizontal retraction toward body center
        self.tuck_retraction_z = 0.12  # vertical retraction toward body center

        # Extension parameters (landing reach)
        self.landing_extension_x_front = 0.06  # front legs reach forward
        self.landing_extension_x_rear = 0.06   # rear legs reach rearward
        self.landing_extension_z = 0.04        # legs extend downward

        # Takeoff velocity parameters
        self.takeoff_vx = 1.2   # forward velocity at takeoff (m/s)
        self.takeoff_vz = 2.5   # upward velocity at takeoff (m/s)

        # Pitch rotation parameters
        self.target_pitch_rotation = 2 * np.pi  # 360 degrees
        self.peak_pitch_rate = 12.0  # rad/s, peak angular velocity mid-flight

    def update_base_motion(self, phase, dt):
        """
        Prescribe base velocity and angular velocity based on phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        roll_rate = 0.0
        yaw_rate = 0.0

        # Phase 0.0-0.2: Crouch/load
        if phase < 0.2:
            # Base descends slowly
            progress = phase / 0.2
            vz = -0.3 * (1.0 - progress)  # downward then zero
            vx = 0.0

        # Phase 0.2-0.35: Explosive takeoff
        elif phase < 0.35:
            progress = (phase - 0.2) / 0.15
            # Smooth ramp using sigmoid-like curve
            s = self._smooth_step(progress)
            vx = self.takeoff_vx * s
            vz = self.takeoff_vz * s
            # Pitch rate ramps up rapidly
            pitch_rate = self.peak_pitch_rate * s

        # Phase 0.35-0.7: Airborne rotation (parabolic trajectory)
        elif phase < 0.7:
            progress = (phase - 0.35) / 0.35
            # Forward velocity decays slightly
            vx = self.takeoff_vx * (1.0 - 0.3 * progress)
            # Vertical velocity: parabolic (up -> peak -> down)
            # Simulate gravity effect kinematically
            t_flight = progress * 0.7  # normalized flight time
            vz = self.takeoff_vz - 9.81 * t_flight * 0.7 / self.freq
            # Pitch rate: peak in middle, then decay
            pitch_profile = np.sin(np.pi * progress)
            pitch_rate = self.peak_pitch_rate * pitch_profile

        # Phase 0.7-0.9: Pre-landing extension (descending)
        elif phase < 0.9:
            progress = (phase - 0.7) / 0.2
            # Forward velocity continues decaying
            vx = self.takeoff_vx * 0.4 * (1.0 - progress)
            # Downward velocity continues
            vz = -2.0 * (1.0 + progress)
            # Pitch rate decays toward zero
            pitch_rate = self.peak_pitch_rate * 0.3 * (1.0 - self._smooth_step(progress))

        # Phase 0.9-1.0: Landing absorption
        else:
            progress = (phase - 0.9) / 0.1
            # Rapid deceleration to zero
            decay = 1.0 - self._smooth_step(progress)
            vx = 0.3 * decay
            vz = -1.5 * decay
            pitch_rate = 0.0

        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])

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
        All legs synchronized.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()

        is_front = leg_name.startswith('F')

        # Phase 0.0-0.2: Crouch/load
        if phase < 0.2:
            progress = self._smooth_step(phase / 0.2)
            foot[2] += self.crouch_depth * progress  # move up toward body (less negative z)
            if is_front:
                foot[0] += self.crouch_forward_shift * progress

        # Phase 0.2-0.35: Explosive takeoff (transition to airborne)
        elif phase < 0.35:
            progress = (phase - 0.2) / 0.15
            s = self._smooth_step(progress)
            # Feet push away then lift off
            crouch_pos = base_pos.copy()
            crouch_pos[2] += self.crouch_depth
            if is_front:
                crouch_pos[0] += self.crouch_forward_shift
            # Interpolate from crouch to slightly extended
            foot = crouch_pos * (1.0 - s) + base_pos * s

        # Phase 0.35-0.7: Airborne tuck
        elif phase < 0.7:
            progress = (phase - 0.35) / 0.35
            s = self._smooth_step(progress)
            # Legs retract toward body center
            tuck_pos = base_pos.copy()
            if is_front:
                tuck_pos[0] -= self.tuck_retraction_x
            else:
                tuck_pos[0] += self.tuck_retraction_x
            tuck_pos[2] += self.tuck_retraction_z
            # Smoothly transition to tuck and hold
            foot = base_pos * (1.0 - s) + tuck_pos * s

        # Phase 0.7-0.9: Pre-landing extension
        elif phase < 0.9:
            progress = (phase - 0.7) / 0.2
            s = self._smooth_step(progress)
            # From tuck to extended landing position
            tuck_pos = base_pos.copy()
            if is_front:
                tuck_pos[0] -= self.tuck_retraction_x
            else:
                tuck_pos[0] += self.tuck_retraction_x
            tuck_pos[2] += self.tuck_retraction_z

            landing_pos = base_pos.copy()
            if is_front:
                landing_pos[0] += self.landing_extension_x_front
            else:
                landing_pos[0] -= self.landing_extension_x_rear
            landing_pos[2] -= self.landing_extension_z

            foot = tuck_pos * (1.0 - s) + landing_pos * s

        # Phase 0.9-1.0: Landing absorption
        else:
            progress = (phase - 0.9) / 0.1
            s = self._smooth_step(progress)
            # From extended landing to compressed stance
            landing_pos = base_pos.copy()
            if is_front:
                landing_pos[0] += self.landing_extension_x_front
            else:
                landing_pos[0] -= self.landing_extension_x_rear
            landing_pos[2] -= self.landing_extension_z

            compressed_pos = base_pos.copy()
            compressed_pos[2] += self.crouch_depth * 0.5  # partial compression

            foot = landing_pos * (1.0 - s) + compressed_pos * s

        return foot

    def _smooth_step(self, x):
        """
        Smoothstep function for continuous transitions.
        Maps [0,1] -> [0,1] with zero derivatives at boundaries.
        """
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)