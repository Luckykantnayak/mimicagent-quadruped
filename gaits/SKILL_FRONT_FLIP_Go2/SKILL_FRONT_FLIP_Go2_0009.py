from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_FRONT_FLIP_Go2_MotionGenerator(BaseMotionGenerator):
    """
    Front flip maneuver: kinematic animation of a 360-degree forward pitch rotation
    with coordinated leg tucking and landing.
    
    Phase structure:
      [0.0, 0.25]: Takeoff and initial rotation - legs push off, body pitches forward
      [0.25, 0.5]: Early aerial rotation - legs tuck in, pitch continues
      [0.5, 0.75]: Inverted rotation - body passes through upside-down, legs remain tucked
      [0.75, 1.0]: Landing preparation - pitch completes 360°, legs extend for touchdown
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for dynamic flip maneuver
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Flip parameters
        self.total_pitch_rotation = 2 * np.pi  # 360 degrees
        self.tuck_distance = 0.15  # How much to retract legs toward body COM
        self.landing_extension = 0.05  # Extra downward extension for landing
        
        # Takeoff parameters
        self.takeoff_vx = 0.8  # Forward velocity during takeoff
        self.takeoff_vz = 1.5  # Upward velocity during takeoff
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on flip phase.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        pitch_rate = 0.0
        
        # Phase 1: Takeoff and initial rotation [0.0, 0.25]
        if phase < 0.25:
            local_phase = phase / 0.25
            vx = self.takeoff_vx * (1.0 - local_phase)  # Reduce forward velocity
            vz = self.takeoff_vz * (1.0 - local_phase * 0.5)  # Start with upward velocity
            # Ramp up pitch rate aggressively
            pitch_rate = 8.0 * np.pi * (0.5 + 0.5 * local_phase)
        
        # Phase 2: Early aerial rotation [0.25, 0.5]
        elif phase < 0.5:
            local_phase = (phase - 0.25) / 0.25
            vx = self.takeoff_vx * 0.5 * (1.0 - local_phase)  # Continue reducing forward velocity
            vz = 0.3 - 1.0 * local_phase  # Transition from upward to downward (apex)
            # Maintain high pitch rate
            pitch_rate = 8.0 * np.pi
        
        # Phase 3: Inverted rotation [0.5, 0.75]
        elif phase < 0.75:
            local_phase = (phase - 0.5) / 0.25
            vx = 0.0  # Minimal forward velocity
            vz = -0.7 - 0.8 * local_phase  # Descending
            # Maintain pitch rate through inversion
            pitch_rate = 8.0 * np.pi * (1.0 - 0.3 * local_phase)
        
        # Phase 4: Landing preparation [0.75, 1.0]
        else:
            local_phase = (phase - 0.75) / 0.25
            vx = 0.0  # Zero forward velocity for stable landing
            vz = -1.5 * (1.0 - local_phase * 0.7)  # Descending until touchdown
            # Decelerate pitch rate to complete exactly 360 degrees
            pitch_rate = 8.0 * np.pi * 0.7 * (1.0 - local_phase)
        
        # Set world frame velocities
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, pitch_rate, 0.0])  # Pitch only
        
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
        Compute foot position in body frame throughout the flip phases.
        Legs tuck during aerial phase and extend for landing.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        is_front = leg_name.startswith('F')
        
        # Phase 1: Takeoff [0.0, 0.1] - push off, then begin retraction [0.1, 0.25]
        if phase < 0.25:
            if phase < 0.1:
                # Push-off: legs extended downward
                local_phase = phase / 0.1
                foot[2] = base_pos[2] - 0.02 * (1.0 - local_phase)  # Slight additional extension
            else:
                # Begin tucking
                local_phase = (phase - 0.1) / 0.15
                tuck_progress = local_phase
                
                # Retract toward body center
                foot[0] = base_pos[0] * (1.0 - 0.5 * tuck_progress)
                foot[1] = base_pos[1] * (1.0 - 0.3 * tuck_progress)
                foot[2] = base_pos[2] + self.tuck_distance * tuck_progress
        
        # Phase 2: Early aerial [0.25, 0.5] - full tuck
        elif phase < 0.5:
            local_phase = (phase - 0.25) / 0.25
            
            # Legs fully tucked, positioned close to body
            foot[0] = base_pos[0] * 0.5
            foot[1] = base_pos[1] * 0.7
            foot[2] = base_pos[2] + self.tuck_distance
        
        # Phase 3: Inverted rotation [0.5, 0.75] - maintain tuck, begin extension
        elif phase < 0.75:
            local_phase = (phase - 0.5) / 0.25
            
            if local_phase < 0.6:
                # Maintain tuck
                foot[0] = base_pos[0] * 0.5
                foot[1] = base_pos[1] * 0.7
                foot[2] = base_pos[2] + self.tuck_distance
            else:
                # Begin extending for landing
                extend_phase = (local_phase - 0.6) / 0.4
                foot[0] = base_pos[0] * (0.5 + 0.5 * extend_phase)
                foot[1] = base_pos[1] * (0.7 + 0.3 * extend_phase)
                foot[2] = base_pos[2] + self.tuck_distance * (1.0 - extend_phase)
        
        # Phase 4: Landing [0.75, 1.0] - extend fully downward
        else:
            local_phase = (phase - 0.75) / 0.25
            
            # Full extension with extra reach for landing
            extension_factor = np.clip(local_phase * 2.0, 0.0, 1.0)
            
            foot[0] = base_pos[0]
            foot[1] = base_pos[1]
            
            # Extend downward with landing extension
            foot[2] = base_pos[2] - self.landing_extension * extension_factor
            
            # Add slight compliance curve near touchdown (phase > 0.85)
            if phase > 0.85:
                contact_phase = (phase - 0.85) / 0.15
                # Simulate impact absorption with small upward deflection
                foot[2] += 0.02 * np.sin(np.pi * contact_phase)
        
        return foot