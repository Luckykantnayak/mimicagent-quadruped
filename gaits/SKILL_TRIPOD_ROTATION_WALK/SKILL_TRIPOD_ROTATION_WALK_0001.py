from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_TRIPOD_ROTATION_WALK_MotionGenerator(BaseMotionGenerator):
    """
    Tripod rotation walk gait with continuous forward motion and clockwise yaw rotation.
    
    - Diagonal pairs alternate: FL+RR swing together, then FR+RL swing together
    - Base maintains constant forward velocity and positive yaw rate throughout
    - Swing trajectories curve in body frame to compensate for continuous rotation
    - Phase [0.0, 0.5]: FL+RR swing, FR+RL stance
    - Phase [0.5, 1.0]: FR+RL swing, FL+RR stance
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Swing parameters
        self.step_length = 0.12  # Forward reach during swing
        self.step_height = 0.06  # Peak clearance height
        self.lateral_curve = 0.03  # Lateral adjustment for yaw compensation
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase offsets for diagonal tripod coordination
        # FL+RR swing during [0.0, 0.5], FR+RL swing during [0.5, 1.0]
        self.phase_offsets = {}
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.phase_offsets[leg] = 0.0  # Group 1: swing first half
            else:  # FR or RL
                self.phase_offsets[leg] = 0.5  # Group 2: swing second half
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Constant velocity commands for tripod rotation walk
        self.vx_forward = 0.15  # Constant forward velocity
        self.yaw_rate_const = 0.8  # Constant clockwise yaw rate

    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward velocity and constant positive yaw rate.
        Both velocities are maintained throughout the entire phase cycle.
        """
        # Constant forward velocity in world frame
        self.vel_world = np.array([self.vx_forward, 0.0, 0.0])
        
        # Constant clockwise (positive) yaw rate
        self.omega_world = np.array([0.0, 0.0, self.yaw_rate_const])
        
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
        Compute foot position in body frame with tripod coordination.
        
        Diagonal pairs alternate:
        - FL+RR: swing during [0.0, 0.5], stance during [0.5, 1.0]
        - FR+RL: stance during [0.0, 0.5], swing during [0.5, 1.0]
        
        Swing trajectories curve to compensate for continuous yaw rotation.
        """
        # Apply phase offset for tripod coordination
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Get base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if left or right leg for lateral curve direction
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        lateral_sign = 1.0 if is_left else -1.0
        
        # Swing phase: [0.0, 0.5) in leg_phase
        # Stance phase: [0.5, 1.0) in leg_phase
        if leg_phase < 0.5:
            # SWING PHASE
            # Progress within swing phase [0, 1]
            swing_progress = leg_phase / 0.5
            
            # Forward motion: arc from rear to front
            # At start (progress=0): foot at rear position
            # At end (progress=1): foot at front position
            foot[0] += self.step_length * (swing_progress - 0.5)
            
            # Lateral curve compensation for yaw rotation
            # Curve peaks at mid-swing and adjusts based on leg side
            curve_progress = np.sin(np.pi * swing_progress)
            foot[1] += lateral_sign * self.lateral_curve * curve_progress
            
            # Vertical swing arc: sinusoidal clearance
            foot[2] += self.step_height * np.sin(np.pi * swing_progress)
            
        else:
            # STANCE PHASE
            # Progress within stance phase [0, 1]
            stance_progress = (leg_phase - 0.5) / 0.5
            
            # During stance, foot remains planted in world frame
            # In body frame, this appears as rearward drift due to forward motion
            # and tangential drift due to yaw rotation
            
            # Rearward drift: foot moves from front to rear in body frame
            foot[0] += self.step_length * (0.5 - stance_progress)
            
            # Tangential drift due to yaw rotation
            # As body rotates clockwise, stance foot appears to drift inward/outward
            tangential_drift = -lateral_sign * self.lateral_curve * stance_progress * 0.5
            foot[1] += tangential_drift
            
            # Foot remains on ground
            foot[2] = self.base_feet_pos_body[leg_name][2]
        
        return foot