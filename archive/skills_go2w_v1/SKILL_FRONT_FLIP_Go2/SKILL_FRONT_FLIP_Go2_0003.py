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
      0.7-0.85: Pre-landing extension - legs extend to reach ground
      0.85-1.0: Landing absorption - all feet contact, legs flex to absorb impact
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

        # Target heights for trajectory management
        self.initial_height = 0.30  # Starting base height
        self.crouch_height = 0.25   # Base height at end of crouch
        self.peak_flight_height = 0.62  # Maximum height during flight
        self.landing_height = 0.42  # Target base height at landing contact (raised from 0.28)

        # Crouch parameters
        self.crouch_depth = 0.10  # meters, vertical retraction during crouch
        self.crouch_forward_shift = 0.02  # front legs shift slightly forward

        # Tuck parameters
        self.tuck_retraction_x = 0.08  # horizontal retraction toward body center
        self.tuck_retraction_z = 0.12  # vertical retraction toward body center

        # Extension parameters (landing reach) - reduced to prevent ground penetration
        self.landing_extension_x_front = 0.06  # front legs reach forward
        self.landing_extension_x_rear = 0.06   # rear legs reach rearward
        self.landing_extension_z = 0.11        # legs extend downward (reduced from 0.18)

        # Takeoff velocity parameters
        self.takeoff_vx = 1.0   # forward velocity at takeoff (m/s)
        self.takeoff_vz = 1.95  # upward velocity at takeoff (m/s) - slightly increased

        # Pitch rotation parameters
        self.target_pitch_rotation = 2 * np.pi  # 360 degrees
        self.peak_pitch_rate = 11.0  # rad/s, peak angular velocity mid-flight

        # Initialize base position to safe starting height
        self.root_pos[2] = self.initial_height

    def update_base_motion(self, phase, dt):
        """
        Prescribe base velocity and angular velocity based on phase.
        Velocities are computed to achieve target heights at phase boundaries.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        roll_rate = 0.0
        yaw_rate = 0.0

        # Phase 0.0-0.2: Crouch/load
        if phase < 0.2:
            progress = phase / 0.2
            s = self._smooth_step(progress)
            # Descend from initial_height to crouch_height
            height_delta = self.crouch_height - self.initial_height
            phase_duration = 0.2 / self.freq
            vz = (height_delta / phase_duration) * (1.0 - s) * 1.5
            vx = 0.0

        # Phase 0.2-0.35: Explosive takeoff
        elif phase < 0.35:
            progress = (phase - 0.2) / 0.15
            s = self._smooth_step(progress)
            vx = self.takeoff_vx * s
            vz = self.takeoff_vz * s
            # Pitch rate ramps up
            pitch_rate = self.peak_pitch_rate * s * 0.6

        # Phase 0.35-0.7: Airborne rotation (parabolic trajectory)
        elif phase < 0.7:
            progress = (phase - 0.35) / 0.35
            # Forward velocity decays slightly
            vx = self.takeoff_vx * (1.0 - 0.2 * progress)
            
            # Vertical velocity: parabolic arc from takeoff to descent
            phase_duration = 0.35 / self.freq
            t_flight = progress * phase_duration
            gravity_equiv = 7.2  # effective kinematic gravity (adjusted from 6.5)
            vz = self.takeoff_vz - gravity_equiv * t_flight
            
            # Pitch rate: peak in middle of rotation
            pitch_profile = np.sin(np.pi * progress)
            pitch_rate = self.peak_pitch_rate * pitch_profile

        # Phase 0.7-0.85: Pre-landing extension (controlled descent)
        elif phase < 0.85:
            progress = (phase - 0.7) / 0.15
            s = self._smooth_step(progress)
            # Forward velocity continues decaying
            vx = self.takeoff_vx * 0.3 * (1.0 - s)
            # Controlled descent to landing height - gentler than before
            vz = -0.7 * (1.0 - 0.5 * s)  # reduced from -1.2
            # Pitch rate completes rotation and decays
            pitch_rate = self.peak_pitch_rate * 0.25 * (1.0 - s)

        # Phase 0.85-1.0: Landing absorption (extended phase duration)
        else:
            progress = (phase - 0.85) / 0.15
            s = self._smooth_step(progress)
            # Rapid deceleration to zero
            decay = 1.0 - s
            vx = 0.15 * decay
            # Gentle final descent with strong damping
            vz = -0.3 * decay
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

        # Enforce height envelope bounds with raised lower limit
        self.root_pos[2] = np.clip(self.root_pos[2], 0.23, 0.65)

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for given leg and phase.
        All legs synchronized. Foot extensions tuned to coordinate with base height trajectory.
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
            # Transition from crouch to neutral, beginning liftoff
            foot = crouch_pos * (1.0 - s) + base_pos * s

        # Phase 0.35-0.7: Airborne tuck
        elif phase < 0.7:
            progress = (phase - 0.35) / 0.35
            s = self._smooth_step(progress)
            # Legs retract toward body center for tuck
            tuck_pos = base_pos.copy()
            if is_front:
                tuck_pos[0] -= self.tuck_retraction_x
            else:
                tuck_pos[0] += self.tuck_retraction_x
            tuck_pos[2] += self.tuck_retraction_z
            # Smoothly transition to tuck position
            foot = base_pos * (1.0 - s) + tuck_pos * s

        # Phase 0.7-0.85: Pre-landing extension
        elif phase < 0.85:
            progress = (phase - 0.7) / 0.15
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
            # Extend downward to reach ground - reduced extension
            landing_pos[2] -= self.landing_extension_z

            foot = tuck_pos * (1.0 - s) + landing_pos * s

        # Phase 0.85-1.0: Landing absorption (extended phase)
        else:
            progress = (phase - 0.85) / 0.15
            s = self._smooth_step(progress)
            # From extended landing to compressed stance
            landing_pos = base_pos.copy()
            if is_front:
                landing_pos[0] += self.landing_extension_x_front
            else:
                landing_pos[0] -= self.landing_extension_x_rear
            landing_pos[2] -= self.landing_extension_z

            # Absorb impact by flexing legs partially, with rapid initial retraction
            compressed_pos = base_pos.copy()
            compressed_pos[2] += self.crouch_depth * 0.5  # increased compression amount

            # Front-load the retraction motion for rapid foot lift
            retraction_curve = self._smooth_step(s) * 1.2
            retraction_curve = np.clip(retraction_curve, 0.0, 1.0)
            
            foot = landing_pos * (1.0 - retraction_curve) + compressed_pos * retraction_curve

        return foot

    def _smooth_step(self, x):
        """
        Smoothstep function for continuous transitions.
        Maps [0,1] -> [0,1] with zero derivatives at boundaries.
        """
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)