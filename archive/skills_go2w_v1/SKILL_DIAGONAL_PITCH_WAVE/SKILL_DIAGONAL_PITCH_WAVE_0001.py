from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_DIAGONAL_PITCH_WAVE_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal pitch wave locomotion.
    
    The robot travels diagonally forward-right via a pitch wave propagating
    along the diagonal body axis (rear-left to front-right). Body rocks in
    a rotated pitch plane aligned with the diagonal direction.
    
    Phase structure:
      [0.0, 0.3]: rear_lift_push - rear lifts, RL/RR push
      [0.3, 0.6]: level_transition - body levels, load handoff
      [0.6, 1.0]: front_lift_pull - front lifts, FL/FR pull
    
    All four legs maintain ground contact throughout (continuous-contact gait).
    Diagonal motion achieved by combining pitch and roll rates to create
    rotation about the FL-RR diagonal axis.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        """
        Initialize diagonal pitch wave motion generator.
        
        Args:
            initial_foot_positions_body: dict mapping leg names to 3D positions in body frame
            leg_names: list of leg names [FL, FR, RL, RR]
        """
        # Store leg configuration
        self.leg_names = leg_names
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Diagonal velocity parameters (forward-right motion)
        self.vx_base = 0.4  # Forward velocity component (m/s)
        self.vy_base = 0.4  # Rightward velocity component (m/s)
        self.vz_amplitude = 0.05  # Vertical oscillation amplitude (m/s)
        
        # Angular velocity parameters for diagonal pitch wave
        # Diagonal pitch achieved by combining pitch and roll
        self.pitch_rate_amplitude = 0.8  # rad/s
        self.roll_rate_amplitude = 0.6  # rad/s
        
        # Leg motion parameters (body frame adjustments)
        self.leg_fore_aft_range = 0.08  # Fore-aft foot motion range in body frame (m)
        self.leg_compression_amplitude = 0.03  # Vertical leg compression/extension (m)
        
        # Phase boundaries
        self.phase_rear_lift_end = 0.3
        self.phase_level_end = 0.6
        
        # Call base class constructor
        BaseMotionGenerator.__init__(self, initial_foot_positions_body, freq=self.freq)

    def update_base_motion(self, phase, dt):
        """
        Update base pose using phase-dependent diagonal pitch wave motion.
        
        Combines linear diagonal velocity with coupled pitch-roll angular velocity
        to create a pitch wave along the diagonal axis.
        """
        # Linear velocity: constant diagonal forward-right with vertical oscillation
        vx = self.vx_base
        vy = self.vy_base
        
        # Vertical velocity component for pitch wave
        if phase < self.phase_rear_lift_end:
            # Rear lift phase: positive vz (CoM rises as rear lifts)
            progress = phase / self.phase_rear_lift_end
            vz = self.vz_amplitude * np.sin(np.pi * progress)
        elif phase < self.phase_level_end:
            # Level transition: vz ~ 0
            vz = 0.0
        else:
            # Front lift phase: negative vz (CoM lowers relative to rear)
            progress = (phase - self.phase_level_end) / (1.0 - self.phase_level_end)
            vz = -self.vz_amplitude * np.sin(np.pi * progress)
        
        self.vel_world = np.array([vx, vy, vz])
        
        # Angular velocity: diagonal pitch wave via coupled pitch and roll
        # Pitch wave propagates from rear-left to front-right
        if phase < self.phase_rear_lift_end:
            # Rear lift phase: positive pitch (rear up), negative roll (right side dips)
            progress = phase / self.phase_rear_lift_end
            pitch_rate = self.pitch_rate_amplitude * np.sin(np.pi * progress)
            roll_rate = -self.roll_rate_amplitude * np.sin(np.pi * progress)
        elif phase < self.phase_level_end:
            # Level transition: rates return to zero
            pitch_rate = 0.0
            roll_rate = 0.0
        else:
            # Front lift phase: negative pitch (front up), positive roll (left side dips)
            progress = (phase - self.phase_level_end) / (1.0 - self.phase_level_end)
            pitch_rate = -self.pitch_rate_amplitude * np.sin(np.pi * progress)
            roll_rate = self.roll_rate_amplitude * np.sin(np.pi * progress)
        
        yaw_rate = 0.0  # No yaw rotation
        
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
        
        All legs remain in contact. Foot trajectories represent passive motion
        relative to body frame as base translates, plus active extension/compression
        coordinated with the pitch wave.
        
        Rear legs (RL, RR): active push during phase 0.0-0.3
        Front legs (FL, FR): active pull during phase 0.6-1.0
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith(('FL', 'FR'))
        is_rear = leg_name.startswith(('RL', 'RR'))
        
        if is_rear:
            # Rear legs: RL, RR
            if phase < self.phase_rear_lift_end:
                # Active push phase: foot moves forward in body frame as leg extends
                progress = phase / self.phase_rear_lift_end
                foot[0] += self.leg_fore_aft_range * np.sin(np.pi * progress)
                foot[2] -= self.leg_compression_amplitude * (1 - np.cos(np.pi * progress))
            elif phase < self.phase_level_end:
                # Transition: foot repositions rearward for next cycle
                progress = (phase - self.phase_rear_lift_end) / (self.phase_level_end - self.phase_rear_lift_end)
                foot[0] += self.leg_fore_aft_range * (1 - progress)
            else:
                # Passive support: foot moves backward as body translates forward
                progress = (phase - self.phase_level_end) / (1.0 - self.phase_level_end)
                foot[0] -= self.leg_fore_aft_range * progress
                foot[2] -= self.leg_compression_amplitude * np.sin(np.pi * progress)
        
        elif is_front:
            # Front legs: FL, FR
            if phase < self.phase_rear_lift_end:
                # Passive support: foot moves backward in body frame
                progress = phase / self.phase_rear_lift_end
                foot[0] -= self.leg_fore_aft_range * progress
                foot[2] += self.leg_compression_amplitude * np.sin(np.pi * progress)
            elif phase < self.phase_level_end:
                # Transition: foot repositions forward to prepare for pull
                progress = (phase - self.phase_rear_lift_end) / (self.phase_level_end - self.phase_rear_lift_end)
                foot[0] -= self.leg_fore_aft_range * (1 - progress)
            else:
                # Active pull phase: foot pulls body forward, shifts rearward in body frame
                progress = (phase - self.phase_level_end) / (1.0 - self.phase_level_end)
                foot[0] -= self.leg_fore_aft_range * (1 - np.sin(np.pi * progress))
                foot[2] += self.leg_compression_amplitude * (1 - np.cos(np.pi * progress))
        
        return foot