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
        
        # Kinematic motion parameters (reduced for envelope compliance)
        self.max_height_gain = 0.25  # Maximum height above initial position
        self.takeoff_vz = 0.4  # Reduced upward velocity
        self.takeoff_vx = 0.5  # Reduced forward velocity
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # Store initial height for reference
        self.initial_height = 0.27

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on flip phase.
        Uses constant pitch rate for exact 360-degree rotation.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        # Constant pitch rate for exact 2π rotation over full cycle
        pitch_rate = 2.0 * np.pi * self.freq * 2.0  # 2π per cycle
        
        # Phase 1: Takeoff and initial rotation [0.0, 0.25]
        if phase < 0.25:
            local_phase = phase / 0.25
            # Smooth takeoff with upward velocity
            vx = self.takeoff_vx * np.cos(np.pi * local_phase / 2.0)
            vz = self.takeoff_vz * np.sin(np.pi * local_phase)
        
        # Phase 2: Early aerial rotation [0.25, 0.5] - reach apex and begin descent
        elif phase < 0.5:
            local_phase = (phase - 0.25) / 0.25
            # Transition through apex
            vx = self.takeoff_vx * 0.3 * (1.0 - local_phase)
            vz = self.takeoff_vz * np.cos(np.pi * local_phase) * 0.8
        
        # Phase 3: Inverted rotation [0.5, 0.75] - descending
        elif phase < 0.75:
            local_phase = (phase - 0.5) / 0.25
            vx = 0.0
            # Controlled descent
            vz = -0.6 * (0.5 + 0.5 * local_phase)
        
        # Phase 4: Landing preparation [0.75, 1.0] - decelerate to ground
        else:
            local_phase = (phase - 0.75) / 0.25
            vx = 0.0
            # Decelerate vertical velocity to zero by end of phase
            descent_profile = np.cos(np.pi * local_phase / 2.0)
            vz = -0.75 * descent_profile
        
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
        
        # Phase 1: Takeoff [0.0, 0.25] - push off and begin retraction
        if phase < 0.25:
            local_phase = phase / 0.25
            
            if local_phase < 0.4:
                # Push-off: legs extended
                push_phase = local_phase / 0.4
                foot[2] = base_pos[2] - 0.01 * np.sin(np.pi * push_phase)
            else:
                # Begin tucking
                tuck_phase = (local_phase - 0.4) / 0.6
                tuck_smooth = 0.5 * (1.0 - np.cos(np.pi * tuck_phase))
                
                foot[0] = base_pos[0] * (1.0 - 0.4 * tuck_smooth)
                foot[1] = base_pos[1] * (1.0 - 0.2 * tuck_smooth)
                foot[2] = base_pos[2] + self.tuck_distance * tuck_smooth * 0.5
        
        # Phase 2: Early aerial [0.25, 0.5] - continue tucking to full
        elif phase < 0.5:
            local_phase = (phase - 0.25) / 0.25
            tuck_smooth = 0.5 + 0.5 * (1.0 - np.cos(np.pi * local_phase))
            
            foot[0] = base_pos[0] * (0.6 - 0.1 * tuck_smooth)
            foot[1] = base_pos[1] * (0.8 - 0.1 * tuck_smooth)
            foot[2] = base_pos[2] + self.tuck_distance * (0.5 + 0.5 * tuck_smooth)
        
        # Phase 3: Inverted rotation [0.5, 0.75] - maintain tuck then begin extension
        elif phase < 0.75:
            local_phase = (phase - 0.5) / 0.25
            
            if local_phase < 0.5:
                # Maintain full tuck
                foot[0] = base_pos[0] * 0.5
                foot[1] = base_pos[1] * 0.7
                foot[2] = base_pos[2] + self.tuck_distance
            else:
                # Begin extending for landing
                extend_phase = (local_phase - 0.5) / 0.5
                extend_smooth = 0.5 * (1.0 - np.cos(np.pi * extend_phase))
                
                foot[0] = base_pos[0] * (0.5 + 0.5 * extend_smooth)
                foot[1] = base_pos[1] * (0.7 + 0.3 * extend_smooth)
                foot[2] = base_pos[2] + self.tuck_distance * (1.0 - extend_smooth)
        
        # Phase 4: Landing [0.75, 1.0] - extend to landing configuration
        else:
            local_phase = (phase - 0.75) / 0.25
            extend_smooth = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            
            # Return to nominal stance
            foot[0] = base_pos[0]
            foot[1] = base_pos[1]
            foot[2] = base_pos[2]
            
            # Add slight compensation for front legs to prevent penetration
            if is_front:
                foot[2] += 0.01 * (1.0 - extend_smooth)
        
        return foot