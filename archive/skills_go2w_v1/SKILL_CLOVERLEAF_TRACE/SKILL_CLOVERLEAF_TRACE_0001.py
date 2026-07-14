from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CLOVERLEAF_TRACE_MotionGenerator(BaseMotionGenerator):
    """
    Cloverleaf trajectory tracing motion with four curved lobes.
    
    The robot traces a four-lobed cloverleaf pattern on the ground through
    coordinated base velocity modulation, yaw control, and synchronized
    diagonal pair leg reaching. Each lobe occupies 0.25 phase duration.
    
    - Lobe 1 (0.0-0.25): Rightward curve with positive yaw
    - Lobe 2 (0.25-0.5): Leftward curve with negative yaw
    - Lobe 3 (0.5-0.75): Forward-right diagonal with positive yaw
    - Lobe 4 (0.75-1.0): Backward-left diagonal with negative yaw
    
    All four feet maintain ground contact throughout. Diagonal pairs
    (FL+RR vs FR+RL) alternate primary support every half-lobe.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time
        self.t = 0.0
        
        # Base state
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity parameters tuned for cloverleaf shape
        # Lobe 1: rightward curve
        self.lobe1_vx = 0.4
        self.lobe1_vy = 0.6
        self.lobe1_yaw_rate = 1.2
        
        # Lobe 2: leftward curve
        self.lobe2_vx_start = 0.3
        self.lobe2_vx_end = -0.2
        self.lobe2_vy_start = 0.5
        self.lobe2_vy_end = -0.6
        self.lobe2_yaw_rate = -1.2
        
        # Lobe 3: forward-right diagonal
        self.lobe3_vx = 0.7
        self.lobe3_vy = 0.4
        self.lobe3_yaw_rate = 1.0
        
        # Lobe 4: backward-left diagonal
        self.lobe4_vx = -0.6
        self.lobe4_vy = -0.4
        self.lobe4_yaw_rate = -1.0
        
        # Roll counterbalancing gain
        self.roll_gain = 0.4
        
        # Leg reaching parameters
        self.reach_amplitude = 0.12  # 12% of nominal stance width
        self.swing_height = 0.03  # Low swing to maintain near-ground contact
        
        # Diagonal pair phase offsets
        # FL and RR are group 1 (phase 0.0)
        # FR and RL are group 2 (phase 0.5)
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0
            else:  # FR or RL
                self.phase_offsets[leg] = 0.5

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on current lobe.
        Smooth transitions at lobe boundaries with velocity ramping.
        """
        vx = 0.0
        vy = 0.0
        yaw_rate = 0.0
        
        # Lobe 1: Rightward curve (phase 0.0 - 0.25)
        if phase < 0.25:
            lobe_progress = phase / 0.25
            vx = self.lobe1_vx * np.sin(np.pi * lobe_progress)
            vy = self.lobe1_vy * np.sin(np.pi * lobe_progress)
            yaw_rate = self.lobe1_yaw_rate * np.sin(np.pi * lobe_progress)
        
        # Lobe 2: Leftward curve (phase 0.25 - 0.5)
        elif phase < 0.5:
            lobe_progress = (phase - 0.25) / 0.25
            # Smooth transition from lobe 1 to lobe 2
            vx = self.lobe2_vx_start * (1 - lobe_progress) + self.lobe2_vx_end * lobe_progress
            vy_mag = self.lobe2_vy_start * (1 - lobe_progress) + self.lobe2_vy_end * lobe_progress
            vy = vy_mag * np.sin(np.pi * lobe_progress)
            yaw_rate = self.lobe2_yaw_rate * np.sin(np.pi * lobe_progress)
        
        # Lobe 3: Forward-right diagonal (phase 0.5 - 0.75)
        elif phase < 0.75:
            lobe_progress = (phase - 0.5) / 0.25
            vx = self.lobe3_vx * np.sin(np.pi * lobe_progress)
            vy = self.lobe3_vy * np.sin(np.pi * lobe_progress)
            yaw_rate = self.lobe3_yaw_rate * np.sin(np.pi * lobe_progress)
        
        # Lobe 4: Backward-left diagonal (phase 0.75 - 1.0)
        else:
            lobe_progress = (phase - 0.75) / 0.25
            vx = self.lobe4_vx * np.sin(np.pi * lobe_progress)
            vy = self.lobe4_vy * np.sin(np.pi * lobe_progress)
            yaw_rate = self.lobe4_yaw_rate * np.sin(np.pi * lobe_progress)
        
        # Roll counterbalancing: oppose lateral acceleration
        roll_rate = -self.roll_gain * (vy + 0.3 * yaw_rate)
        
        # Set world frame velocities
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([roll_rate, 0.0, yaw_rate])
        
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
        Compute foot position in body frame with reaching modulation.
        
        Diagonal pairs alternate between stance and subtle swing every 0.5 phase.
        Reaching amplitude modulates based on lobe requirements:
        - Extended stance during aggressive curves
        - Moderate stance during center crossings
        """
        # Get leg-specific phase
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine current lobe for reaching direction
        lobe = int(phase / 0.25)
        lobe_progress = (phase % 0.25) / 0.25
        
        # Determine if this leg is in stance or swing within its half-cycle
        # Stance: leg_phase 0.0-0.5, Swing: leg_phase 0.5-1.0
        in_stance = leg_phase < 0.5
        
        if in_stance:
            # Stance phase: reaching modulation based on lobe
            stance_progress = leg_phase / 0.5
            
            if lobe == 0:  # Lobe 1: rightward curve
                # Extend right-side legs outward, left-side legs moderate
                if leg_name.startswith('FR') or leg_name.startswith('RR'):
                    foot[1] += self.reach_amplitude * stance_progress  # Reach right
                else:
                    foot[1] -= self.reach_amplitude * 0.5 * stance_progress  # Slight left
                foot[0] += self.reach_amplitude * 0.3 * np.sin(np.pi * stance_progress)  # Forward
            
            elif lobe == 1:  # Lobe 2: leftward curve
                # Extend left-side legs outward, right-side legs moderate
                if leg_name.startswith('FL') or leg_name.startswith('RL'):
                    foot[1] -= self.reach_amplitude * stance_progress  # Reach left
                else:
                    foot[1] += self.reach_amplitude * 0.5 * stance_progress  # Slight right
                foot[0] -= self.reach_amplitude * 0.2 * np.sin(np.pi * stance_progress)  # Slight back
            
            elif lobe == 2:  # Lobe 3: forward-right diagonal
                # Extend forward and right
                foot[0] += self.reach_amplitude * stance_progress  # Forward
                foot[1] += self.reach_amplitude * 0.6 * stance_progress  # Right
            
            else:  # Lobe 4: backward-left diagonal
                # Extend backward and left
                foot[0] -= self.reach_amplitude * stance_progress  # Backward
                foot[1] -= self.reach_amplitude * 0.6 * stance_progress  # Left
        
        else:
            # Swing phase: low-amplitude repositioning
            swing_progress = (leg_phase - 0.5) / 0.5
            
            # Small arc trajectory for repositioning
            swing_angle = np.pi * swing_progress
            foot[2] += self.swing_height * np.sin(swing_angle)  # Vertical arc
            
            # Horizontal repositioning depends on upcoming lobe
            next_lobe = (lobe + 1) % 4
            
            if next_lobe == 0:  # Preparing for lobe 1: rightward
                if leg_name.startswith('FR') or leg_name.startswith('RR'):
                    foot[1] += self.reach_amplitude * 0.5 * swing_progress
                foot[0] += self.reach_amplitude * 0.2 * swing_progress
            
            elif next_lobe == 1:  # Preparing for lobe 2: leftward
                if leg_name.startswith('FL') or leg_name.startswith('RL'):
                    foot[1] -= self.reach_amplitude * 0.5 * swing_progress
            
            elif next_lobe == 2:  # Preparing for lobe 3: forward-right
                foot[0] += self.reach_amplitude * 0.5 * swing_progress
                foot[1] += self.reach_amplitude * 0.3 * swing_progress
            
            else:  # Preparing for lobe 4: backward-left
                foot[0] -= self.reach_amplitude * 0.5 * swing_progress
                foot[1] -= self.reach_amplitude * 0.3 * swing_progress
        
        return foot