from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERSE_CARTWHEEL_DRIFT_MotionGenerator(BaseMotionGenerator):
    """
    Reverse cartwheel drift: continuous backward locomotion via 360-degree body roll cycles.
    
    Motion characteristics:
    - Continuous positive roll rotation (full cartwheel per cycle)
    - Sustained backward drift (negative x velocity)
    - Left legs (FL, RL) contact ground phase 0.0-0.2, push to generate drift
    - Right legs (FR, RR) contact ground phase 0.65-1.0, stabilize landing
    - Airborne inverted phase 0.2-0.65 with full leg extension
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.  # Slower cycle for dynamic cartwheel motion
        
        # Base foot positions (body frame reference)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Motion parameters
        self.backward_speed = -0.4  # Sustained backward drift velocity
        self.roll_rate = 6.6  # Continuous positive roll rate (rad/s) for 360° rotation
        
        # Leg motion parameters
        self.radial_extension = 0.25  # Maximum radial extension during cartwheel
        self.push_depth = 0.08  # Vertical push depth during stance
        self.backward_push = 0.12  # Horizontal backward displacement during push

    def update_base_motion(self, phase, dt):
        """
        Base motion: continuous backward drift + continuous positive roll.
        Phase-modulated vertical and roll rate adjustments for realistic dynamics.
        """
        # Sustained backward drift throughout cycle
        vx = self.backward_speed
        
        # Modulate vertical velocity per phase
        if phase < 0.25:
            # Right side ascent: slight upward for push-off
            vz = 0.15
        elif phase < 0.5:
            # Inverted flight: ballistic (neutral)
            vz = 0.0
        elif phase < 0.75:
            # Left side descent: slight downward for landing prep
            vz = -0.15
        else:
            # Upright return: neutral
            vz = 0.0
        
        # Continuous positive roll rate throughout entire cycle
        roll_rate = self.roll_rate
        
        # Apply velocity commands
        self.vel_world = np.array([vx, 0.0, vz])
        self.omega_world = np.array([roll_rate, 0.0, 0.0])
        
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
        Compute foot trajectory in body frame per leg and phase.
        
        FL/RL (left legs):
          - Phase 0.0-0.2: grounded stance, push backward and down
          - Phase 0.2-0.65: swing overhead with radial extension
          - Phase 0.65-1.0: retract and return to nominal position
        
        FR/RR (right legs):
          - Phase 0.0-0.2: lift and begin radial extension
          - Phase 0.2-0.65: overhead extension, then descend
          - Phase 0.65-1.0: grounded stance, absorb landing
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Identify leg side
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        if is_left:
            # Left legs: FL, RL
            if phase < 0.2:
                # Stance phase: grounded push
                progress = phase / 0.2
                foot = base_pos.copy()
                # Push backward
                foot[0] -= self.backward_push * progress
                # Push down (negative z in body frame = extension down)
                foot[2] -= self.push_depth * np.sin(np.pi * progress)
            
            elif phase < 0.65:
                # Swing phase: radial extension overhead during cartwheel
                progress = (phase - 0.2) / (0.65 - 0.2)
                angle = np.pi * progress  # 0 to π
                
                # Radial extension trajectory (circular arc in body frame)
                # Legs extend outward and trace overhead arc
                foot = base_pos.copy()
                foot[0] -= self.backward_push  # Maintain backward offset
                
                # Radial extension: outward in local y-z plane
                # At progress=0.5 (phase≈0.425), maximum extension (inverted)
                extension = self.radial_extension * np.sin(angle)
                foot[1] += np.sign(base_pos[1]) * extension * 0.5  # Lateral extension
                foot[2] += extension  # Upward extension in body frame
            
            else:
                # Recovery phase: retract and return to nominal
                progress = (phase - 0.65) / (1.0 - 0.65)
                foot = base_pos.copy()
                # Smooth return with sinusoidal easing
                offset_x = self.backward_push * (1.0 - progress)
                foot[0] -= offset_x
        
        elif is_right:
            # Right legs: FR, RR
            if phase < 0.2:
                # Initial lift and extension start
                progress = phase / 0.2
                foot = base_pos.copy()
                # Begin radial extension upward
                extension = self.radial_extension * progress
                foot[1] += np.sign(base_pos[1]) * extension * 0.3
                foot[2] += extension * 0.5
            
            elif phase < 0.65:
                # Overhead extension, then descent
                progress = (phase - 0.2) / (0.65 - 0.2)
                angle = np.pi * progress
                
                foot = base_pos.copy()
                
                # Full overhead extension during first half, then descend
                if progress < 0.5:
                    # Ascending to full extension
                    sub_progress = progress / 0.5
                    extension = self.radial_extension * sub_progress
                    foot[1] += np.sign(base_pos[1]) * extension * 0.5
                    foot[2] += extension
                else:
                    # Descending from extension toward landing
                    sub_progress = (progress - 0.5) / 0.5
                    extension = self.radial_extension * (1.0 - sub_progress)
                    foot[1] += np.sign(base_pos[1]) * extension * 0.5
                    foot[2] += extension * 0.5
            
            else:
                # Stance phase: grounded, absorb landing and stabilize
                progress = (phase - 0.65) / (1.0 - 0.65)
                foot = base_pos.copy()
                # Slight compression during landing absorption
                compression = self.push_depth * 0.5 * np.sin(np.pi * progress)
                foot[2] -= compression
        
        else:
            # Fallback (should not occur with standard leg names)
            foot = base_pos.copy()
        
        return foot