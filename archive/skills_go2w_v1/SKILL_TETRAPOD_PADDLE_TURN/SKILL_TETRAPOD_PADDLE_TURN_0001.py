from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_TETRAPOD_PADDLE_TURN_MotionGenerator(BaseMotionGenerator):
    """
    Tetrapod paddle turn: in-place yaw rotation via sequential leg paddling.
    
    Each leg lifts and sweeps through a wide outward arc in sequence:
    FL [0.0, 0.25] → FR [0.25, 0.5] → RR [0.5, 0.75] → RL [0.75, 1.0]
    
    While one leg paddles, the other three anchor firmly to provide stable support
    and reaction forces. The paddle sweeps generate continuous rightward (clockwise)
    yaw rotation with minimal translation.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz) - one full paddle sequence per cycle
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Paddle motion parameters
        self.paddle_lift_height = 0.05  # Minimal lift during paddle sweep
        self.paddle_lateral_amplitude = 0.15  # Lateral outward displacement during sweep
        self.paddle_longitudinal_amplitude = 0.12  # Forward/backward displacement during sweep
        
        # Base motion parameters
        self.yaw_rate_magnitude = 1.2  # rad/s - tuned for ~30 degree rotation per cycle
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (world frame)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base using continuous yaw rotation with near-zero translation.
        
        Yaw rate is constant and positive (rightward rotation) throughout all phases.
        Linear velocities are near-zero with small corrections to counteract paddle drag.
        """
        # Small linear velocity corrections based on which leg is paddling
        vx = 0.0
        vy = 0.0
        
        if phase < 0.25:
            # FL paddling forward-outward: slight forward bias to counteract drag
            vx = 0.02
        elif phase < 0.5:
            # FR paddling forward-outward: slight forward bias
            vx = 0.02
        elif phase < 0.75:
            # RR paddling backward-outward: slight rearward bias
            vx = -0.02
        else:
            # RL paddling backward-outward: slight rearward bias
            vx = -0.02
        
        # Continuous positive yaw rate for clockwise rotation
        yaw_rate = self.yaw_rate_magnitude
        
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate pose in world frame
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
        
        Each leg executes a paddle sweep during its designated quarter-phase window,
        then remains anchored for the remaining three quarters.
        
        Paddle trajectory: lift slightly, sweep wide outward arc (forward for front legs,
        backward for rear legs), emphasizing lateral displacement for yaw torque.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this leg is currently paddling and its local phase
        is_paddling = False
        local_phase = 0.0
        
        if leg_name.startswith('FL'):
            # FL paddles during [0.0, 0.25]
            if phase < 0.25:
                is_paddling = True
                local_phase = phase / 0.25
        elif leg_name.startswith('FR'):
            # FR paddles during [0.25, 0.5]
            if 0.25 <= phase < 0.5:
                is_paddling = True
                local_phase = (phase - 0.25) / 0.25
        elif leg_name.startswith('RR'):
            # RR paddles during [0.5, 0.75]
            if 0.5 <= phase < 0.75:
                is_paddling = True
                local_phase = (phase - 0.5) / 0.25
        elif leg_name.startswith('RL'):
            # RL paddles during [0.75, 1.0]
            if 0.75 <= phase:
                is_paddling = True
                local_phase = (phase - 0.75) / 0.25
        
        if is_paddling:
            # Execute paddle sweep trajectory
            foot = self._compute_paddle_trajectory(leg_name, base_pos, local_phase)
        else:
            # Anchor stance: maintain stable contact position
            # Gradually drift back toward nominal position as base rotates
            foot = self._compute_anchor_position(leg_name, base_pos, phase)
        
        return foot

    def _compute_paddle_trajectory(self, leg_name, base_pos, local_phase):
        """
        Compute paddle sweep trajectory for active leg.
        
        Trajectory: smooth arc with lift, lateral outward sweep, and longitudinal motion.
        Front legs sweep forward-outward, rear legs sweep backward-outward.
        
        local_phase ∈ [0, 1] over the paddle quarter-phase window.
        """
        foot = base_pos.copy()
        
        # Smooth swing profile using sinusoidal blending
        swing_progress = local_phase
        
        # Lift trajectory: minimal height, smooth up and down
        lift_angle = np.pi * swing_progress
        z_offset = self.paddle_lift_height * np.sin(lift_angle)
        
        # Lateral (y-axis) sweep: outward arc
        # Positive y for left legs (FL, RL), negative y for right legs (FR, RR)
        lateral_sign = 1.0 if leg_name.startswith('FL') or leg_name.startswith('RL') else -1.0
        y_offset = lateral_sign * self.paddle_lateral_amplitude * np.sin(lift_angle)
        
        # Longitudinal (x-axis) sweep: forward for front legs, backward for rear legs
        if leg_name.startswith('FL') or leg_name.startswith('FR'):
            # Front legs: sweep forward
            x_offset = self.paddle_longitudinal_amplitude * (swing_progress - 0.5)
        else:
            # Rear legs: sweep backward
            x_offset = -self.paddle_longitudinal_amplitude * (swing_progress - 0.5)
        
        foot[0] += x_offset
        foot[1] += y_offset
        foot[2] += z_offset
        
        return foot

    def _compute_anchor_position(self, leg_name, base_pos, phase):
        """
        Compute anchor stance position for non-paddling leg.
        
        Anchor legs maintain firm contact with gradual drift back toward nominal
        position as the base rotates beneath them. This simulates the kinematic
        effect of the body rotating while feet stay planted.
        """
        foot = base_pos.copy()
        
        # Determine how far into the anchor phase this leg is
        # Anchor phase is 0.75 of the full cycle (three quarters)
        if leg_name.startswith('FL'):
            # FL anchors during [0.25, 1.0]
            if phase >= 0.25:
                anchor_progress = (phase - 0.25) / 0.75
            else:
                anchor_progress = 0.0
        elif leg_name.startswith('FR'):
            # FR anchors during [0.5, 1.0] and [0.0, 0.25]
            if phase >= 0.5:
                anchor_progress = (phase - 0.5) / 0.5
            elif phase < 0.25:
                anchor_progress = 0.5 + phase / 0.25 * 0.5
            else:
                anchor_progress = 0.0
        elif leg_name.startswith('RR'):
            # RR anchors during [0.75, 1.0] and [0.0, 0.5]
            if phase >= 0.75:
                anchor_progress = (phase - 0.75) / 0.25
            elif phase < 0.5:
                anchor_progress = 0.25 + phase / 0.5 * 0.75
            else:
                anchor_progress = 0.0
        elif leg_name.startswith('RL'):
            # RL anchors during [0.0, 0.75]
            if phase < 0.75:
                anchor_progress = phase / 0.75
            else:
                anchor_progress = 0.0
        
        # Small retraction during anchor phase to prepare for next cycle
        # This creates a smooth transition and compensates for base rotation
        retraction_factor = 0.05 * anchor_progress
        
        # Slight inward and centering drift
        foot[0] *= (1.0 - retraction_factor * 0.3)
        foot[1] *= (1.0 - retraction_factor * 0.2)
        
        return foot