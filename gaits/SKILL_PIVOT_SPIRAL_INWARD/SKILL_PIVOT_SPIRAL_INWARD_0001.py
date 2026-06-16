from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_PIVOT_SPIRAL_INWARD_MotionGenerator(BaseMotionGenerator):
    """
    Inward spiral motion: robot executes a continuously tightening spiral,
    rotating counterclockwise while radius decreases from wide arc to near-spin.
    
    - Left legs (FL, RL) act as inner pivot with minimal displacement
    - Right legs (FR, RR) trace progressively shrinking outer arc
    - Linear velocity decreases while yaw rate increases over phase [0,1]
    - Diagonal gait pattern maintains at least three feet in contact
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for smooth spiral execution
        self.duty = 0.75  # Extended stance for stability during turns

        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Diagonal gait phase offsets
        # FL and RR swing together (group 1), FR and RL swing together (group 2)
        self.phase_offsets = {
            leg_names[0]: 0.0,   # FL
            leg_names[1]: 0.5,   # FR
            leg_names[2]: 0.5,   # RL
            leg_names[3]: 0.0,   # RR
        }

        # Swing parameters
        self.step_height = 0.06  # Moderate swing height

        # Left legs (pivot): minimal swing displacement
        self.left_step_length = 0.02
        self.left_lateral_offset = -0.01  # Slightly inward

        # Right legs (outer arc): progressively decreasing swing displacement
        self.right_step_length_initial = 0.15
        self.right_step_length_final = 0.03
        self.right_lateral_offset_initial = 0.08
        self.right_lateral_offset_final = 0.02

        # Base velocity parameters
        self.vx_initial = 0.4
        self.vx_final = 0.05
        self.vy_initial = 0.3
        self.vy_final = 0.02
        self.yaw_rate_initial = 0.6
        self.yaw_rate_final = 2.5

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base motion with decreasing linear velocity and increasing yaw rate.
        Creates inward spiral trajectory.
        """
        # Smooth interpolation using cubic easing for natural spiral tightening
        ease = phase ** 2

        # Linear velocities decrease as spiral tightens
        vx = self.vx_initial + (self.vx_final - self.vx_initial) * ease
        vy = self.vy_initial + (self.vy_final - self.vy_initial) * ease

        # Yaw rate increases as spiral tightens (counterclockwise = positive)
        yaw_rate = self.yaw_rate_initial + (self.yaw_rate_final - self.yaw_rate_initial) * ease

        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])

        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot trajectory in BODY frame.
        
        Left legs (FL, RL): minimal displacement, act as pivot
        Right legs (FR, RR): progressively decreasing arc radius with phase
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()

        # Determine if left or right leg
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')

        if is_left:
            # Left legs: minimal displacement pivot behavior
            step_length = self.left_step_length
            lateral_offset = self.left_lateral_offset

            if leg_phase < self.duty:
                # Stance: minimal backward sweep
                progress = leg_phase / self.duty
                foot[0] -= step_length * (progress - 0.5)
                foot[1] += lateral_offset
            else:
                # Swing: short, low arc
                progress = (leg_phase - self.duty) / (1 - self.duty)
                angle = np.pi * progress
                foot[0] += step_length * (progress - 0.5)
                foot[1] += lateral_offset
                foot[2] += self.step_height * np.sin(angle) * 0.7  # Lower swing

        else:
            # Right legs: progressively decreasing arc radius
            # Interpolate step parameters based on global phase (spiral tightening)
            ease = phase ** 2
            step_length = self.right_step_length_initial + \
                         (self.right_step_length_final - self.right_step_length_initial) * ease
            lateral_offset = self.right_lateral_offset_initial + \
                            (self.right_lateral_offset_final - self.right_lateral_offset_initial) * ease

            if leg_phase < self.duty:
                # Stance: backward sweep with outward offset that decreases over phase
                progress = leg_phase / self.duty
                foot[0] -= step_length * (progress - 0.5)
                foot[1] += lateral_offset * (1.0 - progress * 0.3)  # Retract inward during stance
            else:
                # Swing: arc trajectory with decreasing radius
                progress = (leg_phase - self.duty) / (1 - self.duty)
                angle = np.pi * progress
                foot[0] += step_length * (progress - 0.5)
                foot[1] += lateral_offset * (1.0 - 0.2 * progress)  # Arc inward during swing
                foot[2] += self.step_height * np.sin(angle)

        return foot