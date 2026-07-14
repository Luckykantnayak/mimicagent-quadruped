from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_ZIPPER_MERGE_LATERAL_MotionGenerator(BaseMotionGenerator):
    """
    Zipper merge lateral gait: sequential single-leg stepping with interleaving pattern.
    
    Legs step in sequence FL → FR → RL → RR, each moving laterally left and creating
    a woven cross-pattern. Phase [0.8-1.0] consolidates all legs into merged configuration
    then spreads back to initial stance width.
    
    - Base translates steadily left during stepping phases [0.0-0.8]
    - Roll and pitch modulate to shift weight over support triangle
    - All legs contact during consolidation phase [0.8-1.0]
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Slower cycle for complex sequential coordination
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Gait parameters
        self.step_height = 0.06  # Moderate lift for clearance during interleaving
        self.lateral_step_distance = 0.15  # Leftward displacement per leg step
        self.merge_distance = 0.08  # How far inward legs pull during consolidation
        
        # Base velocity parameters
        self.lateral_velocity = -0.25  # Sustained leftward velocity (negative y)
        self.base_height_oscillation = 0.02  # Slight vertical bobbing during swings
        
        # Roll and pitch modulation amplitudes (subtle weight shifting)
        self.roll_amp = 0.08  # radians
        self.pitch_amp = 0.06  # radians
        
        # Time and state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track leg-specific lateral offsets (accumulate during cycle)
        self.leg_lateral_offsets = {leg: 0.0 for leg in leg_names}

    def update_base_motion(self, phase, dt):
        """
        Update base pose with lateral translation and subtle roll/pitch modulation.
        
        Phase [0.0-0.8]: Sustained leftward velocity with roll/pitch oscillations
        Phase [0.8-1.0]: Decelerate lateral motion and center orientation
        """
        
        if phase < 0.8:
            # Sustained leftward motion during stepping phases
            vy = self.lateral_velocity
            
            # Slight vertical bobbing to aid leg clearance
            # Peak during mid-swing phases
            swing_phase_indicator = np.sin(2 * np.pi * phase * 4)  # 4 swings per cycle
            vz = self.base_height_oscillation * swing_phase_indicator * 0.5
            
            # Roll modulation: shift weight as support triangle changes
            # Positive roll when right legs swing, negative when left legs swing
            # FL swing [0.0-0.2], FR swing [0.2-0.4], RL swing [0.4-0.6], RR swing [0.6-0.8]
            if phase < 0.2:
                # FL swinging: roll right to shift weight onto FR, RL, RR
                roll_rate = self.roll_amp * np.cos(2 * np.pi * phase / 0.2)
            elif phase < 0.4:
                # FR swinging: roll left to shift weight onto FL, RL, RR
                roll_rate = -self.roll_amp * np.cos(2 * np.pi * (phase - 0.2) / 0.2)
            elif phase < 0.6:
                # RL swinging: roll right to shift weight onto FL, FR, RR
                roll_rate = self.roll_amp * np.cos(2 * np.pi * (phase - 0.4) / 0.2)
            else:
                # RR swinging: roll left to shift weight onto FL, FR, RL
                roll_rate = -self.roll_amp * np.cos(2 * np.pi * (phase - 0.6) / 0.2)
            
            # Pitch modulation: balance as front vs rear legs swing
            # Pitch forward when rear legs swing, backward when front legs swing
            if phase < 0.4:
                # Front legs swinging: pitch backward slightly
                pitch_rate = -self.pitch_amp * np.sin(2 * np.pi * phase / 0.4)
            else:
                # Rear legs swinging: pitch forward slightly
                pitch_rate = self.pitch_amp * np.sin(2 * np.pi * (phase - 0.4) / 0.4)
            
            yaw_rate = 0.0
            
        else:
            # Consolidation phase [0.8-1.0]: decelerate and stabilize
            phase_local = (phase - 0.8) / 0.2
            
            # Decay lateral velocity to near zero
            vy = self.lateral_velocity * (1.0 - phase_local)
            
            # Settle downward then stabilize
            vz = -self.base_height_oscillation * (1.0 - phase_local) * 0.5
            
            # Center roll and pitch
            roll_rate = 0.0
            pitch_rate = 0.0
            yaw_rate = 0.0
        
        # Set velocity commands
        self.vel_world = np.array([0.0, vy, vz])
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
        Compute foot position in body frame based on phase and leg-specific sequence.
        
        Sequential swing pattern:
        - FL: [0.0-0.2] swing left
        - FR: [0.2-0.4] swing left + inward (interleave)
        - RL: [0.4-0.6] swing left (interleave)
        - RR: [0.6-0.8] swing left to complete merge
        - All: [0.8-1.0] consolidate inward then spread outward
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Identify leg side for interleaving logic
        is_left_leg = leg_name.startswith("FL") or leg_name.startswith("RL")
        is_front_leg = leg_name.startswith("FL") or leg_name.startswith("FR")
        
        # FL: swing [0.0-0.2]
        if leg_name.startswith("FL"):
            if phase < 0.2:
                # Swing phase: arc leftward
                progress = phase / 0.2
                angle = np.pi * progress
                foot[1] -= self.lateral_step_distance * progress  # Move left (negative y)
                foot[2] += self.step_height * np.sin(angle)  # Lift for clearance
            elif phase < 0.8:
                # Stance phase: remain at extended left position
                foot[1] -= self.lateral_step_distance
            else:
                # Consolidation phase [0.8-1.0]
                phase_local = (phase - 0.8) / 0.2
                # First half: merge inward
                if phase_local < 0.5:
                    merge_progress = phase_local / 0.5
                    foot[1] -= self.lateral_step_distance * (1.0 - merge_progress * 0.5)
                    foot[1] += self.merge_distance * merge_progress
                # Second half: spread back to initial
                else:
                    spread_progress = (phase_local - 0.5) / 0.5
                    foot[1] -= self.lateral_step_distance * 0.5 * (1.0 - spread_progress)
                    foot[1] += self.merge_distance * (1.0 - spread_progress)
        
        # FR: swing [0.2-0.4]
        elif leg_name.startswith("FR"):
            if phase < 0.2:
                # Stance phase: initial position
                pass
            elif phase < 0.4:
                # Swing phase: arc leftward and inward to interleave
                progress = (phase - 0.2) / 0.2
                angle = np.pi * progress
                foot[1] -= self.lateral_step_distance * progress  # Move left
                foot[1] += self.merge_distance * 0.3 * progress  # Slight inward bias for interleaving
                foot[2] += self.step_height * np.sin(angle)  # Lift
            elif phase < 0.8:
                # Stance phase: remain at interleaved position
                foot[1] -= self.lateral_step_distance
                foot[1] += self.merge_distance * 0.3
            else:
                # Consolidation phase [0.8-1.0]
                phase_local = (phase - 0.8) / 0.2
                if phase_local < 0.5:
                    merge_progress = phase_local / 0.5
                    current_y = -self.lateral_step_distance + self.merge_distance * 0.3
                    foot[1] += current_y * (1.0 - merge_progress * 0.5)
                    foot[1] += self.merge_distance * merge_progress
                else:
                    spread_progress = (phase_local - 0.5) / 0.5
                    current_y = -self.lateral_step_distance + self.merge_distance * 0.3
                    foot[1] += current_y * 0.5 * (1.0 - spread_progress)
                    foot[1] += self.merge_distance * (1.0 - spread_progress)
        
        # RL: swing [0.4-0.6]
        elif leg_name.startswith("RL"):
            if phase < 0.4:
                # Stance phase: initial position
                pass
            elif phase < 0.6:
                # Swing phase: arc leftward, threading between right-side legs
                progress = (phase - 0.4) / 0.2
                angle = np.pi * progress
                foot[1] -= self.lateral_step_distance * progress  # Move left
                foot[2] += self.step_height * np.sin(angle)  # Lift
            elif phase < 0.8:
                # Stance phase: remain at interleaved left position
                foot[1] -= self.lateral_step_distance
            else:
                # Consolidation phase [0.8-1.0]
                phase_local = (phase - 0.8) / 0.2
                if phase_local < 0.5:
                    merge_progress = phase_local / 0.5
                    foot[1] -= self.lateral_step_distance * (1.0 - merge_progress * 0.5)
                    foot[1] += self.merge_distance * merge_progress
                else:
                    spread_progress = (phase_local - 0.5) / 0.5
                    foot[1] -= self.lateral_step_distance * 0.5 * (1.0 - spread_progress)
                    foot[1] += self.merge_distance * (1.0 - spread_progress)
        
        # RR: swing [0.6-0.8]
        elif leg_name.startswith("RR"):
            if phase < 0.6:
                # Stance phase: initial position
                pass
            elif phase < 0.8:
                # Swing phase: arc leftward to complete merge
                progress = (phase - 0.6) / 0.2
                angle = np.pi * progress
                foot[1] -= self.lateral_step_distance * progress  # Move left
                foot[2] += self.step_height * np.sin(angle)  # Lift
            else:
                # Consolidation phase [0.8-1.0]
                phase_local = (phase - 0.8) / 0.2
                if phase_local < 0.5:
                    merge_progress = phase_local / 0.5
                    foot[1] -= self.lateral_step_distance * (1.0 - merge_progress * 0.5)
                    foot[1] += self.merge_distance * merge_progress
                else:
                    spread_progress = (phase_local - 0.5) / 0.5
                    foot[1] -= self.lateral_step_distance * 0.5 * (1.0 - spread_progress)
                    foot[1] += self.merge_distance * (1.0 - spread_progress)
        
        return foot