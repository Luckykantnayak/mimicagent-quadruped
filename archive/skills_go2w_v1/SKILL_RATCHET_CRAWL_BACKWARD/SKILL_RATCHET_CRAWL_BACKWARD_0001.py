from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_RATCHET_CRAWL_BACKWARD_MotionGenerator(BaseMotionGenerator):
    """
    Ratchet crawl backward gait.
    
    - All four legs maintain ground contact throughout the cycle
    - Front and rear leg pairs alternate between locked stance and sliding backward
    - Base moves backward in discrete ratcheting steps through four lock-slide sub-phases
    - Neutral transition phase at end smooths cycle repetition
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for controlled ratcheting motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Ratchet motion parameters
        self.slide_distance = 0.12  # Distance legs slide backward in body frame per ratchet
        self.base_backward_speed = -0.3  # Negative x velocity for backward motion
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base motion with backward velocity during ratchet phases.
        Base moves backward during phases 0.0-0.8, decelerates during neutral phase 0.8-1.0.
        """
        if phase < 0.8:
            # Active ratcheting phases - constant backward velocity
            vx = self.base_backward_speed
        else:
            # Neutral transition phase - smooth deceleration to zero
            transition_progress = (phase - 0.8) / 0.2
            vx = self.base_backward_speed * (1.0 - transition_progress)

        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, 0.0])

        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame based on ratchet crawl pattern.
        
        Front legs (FL, FR):
            - [0.0, 0.2]: locked at forward position
            - [0.2, 0.4]: slide backward smoothly
            - [0.4, 0.6]: locked at forward position
            - [0.6, 0.8]: slide backward smoothly
            - [0.8, 1.0]: return to neutral
            
        Rear legs (RL, RR):
            - [0.0, 0.2]: slide backward smoothly
            - [0.2, 0.4]: locked at forward position
            - [0.4, 0.6]: slide backward smoothly
            - [0.6, 0.8]: locked at forward position
            - [0.8, 1.0]: return to neutral
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        
        if is_front:
            # Front legs: locked at [0.0, 0.2] and [0.4, 0.6], slide at [0.2, 0.4] and [0.6, 0.8]
            if 0.0 <= phase < 0.2:
                # Locked at forward position
                foot[0] += self.slide_distance * 0.5
            elif 0.2 <= phase < 0.4:
                # Slide backward smoothly
                slide_progress = (phase - 0.2) / 0.2
                foot[0] += self.slide_distance * (0.5 - slide_progress)
            elif 0.4 <= phase < 0.6:
                # Locked at forward position
                foot[0] += self.slide_distance * 0.5
            elif 0.6 <= phase < 0.8:
                # Slide backward smoothly
                slide_progress = (phase - 0.6) / 0.2
                foot[0] += self.slide_distance * (0.5 - slide_progress)
            else:
                # Neutral transition: return to centered stance
                transition_progress = (phase - 0.8) / 0.2
                current_offset = self.slide_distance * (-0.5)
                foot[0] += current_offset * (1.0 - transition_progress)
        else:
            # Rear legs: slide at [0.0, 0.2] and [0.4, 0.6], locked at [0.2, 0.4] and [0.6, 0.8]
            if 0.0 <= phase < 0.2:
                # Slide backward smoothly
                slide_progress = phase / 0.2
                foot[0] += self.slide_distance * (0.5 - slide_progress)
            elif 0.2 <= phase < 0.4:
                # Locked at forward position
                foot[0] += self.slide_distance * 0.5
            elif 0.4 <= phase < 0.6:
                # Slide backward smoothly
                slide_progress = (phase - 0.4) / 0.2
                foot[0] += self.slide_distance * (0.5 - slide_progress)
            elif 0.6 <= phase < 0.8:
                # Locked at forward position
                foot[0] += self.slide_distance * 0.5
            else:
                # Neutral transition: return to centered stance
                transition_progress = (phase - 0.8) / 0.2
                current_offset = self.slide_distance * 0.5
                foot[0] += current_offset * (1.0 - transition_progress)
        
        # All feet remain at ground level (no vertical motion)
        return foot