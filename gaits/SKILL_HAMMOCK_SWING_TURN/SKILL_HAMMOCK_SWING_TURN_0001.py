from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_HAMMOCK_SWING_TURN_MotionGenerator(BaseMotionGenerator):
    """
    Hammock swing turn: circular path with superimposed lateral oscillation.
    
    - Base executes continuous forward velocity + yaw rate for circular trajectory
    - Lateral velocity oscillates sinusoidally (~2.5 cycles per rotation) for hammock sway
    - All four feet maintain continuous ground contact throughout
    - Feet adjust positions in body frame to track compound base motion
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 1.0

        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Circular path parameters
        self.forward_velocity = 0.4
        self.yaw_rate = 0.8

        # Lateral sway parameters
        self.lateral_sway_amplitude = 0.15
        self.lateral_sway_frequency = 2.5

        # Foot shuffle parameters for continuous contact adjustment
        self.shuffle_amplitude_x = 0.08
        self.shuffle_amplitude_y = 0.06
        
        # Diagonal pair phase offsets for micro-adjustments
        self.phase_offsets = {
            leg_names[0]: 0.0,   # FL
            leg_names[1]: 0.5,   # FR
            leg_names[2]: 0.5,   # RL
            leg_names[3]: 0.0,   # RR
        }

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity, constant yaw rate,
        and sinusoidal lateral velocity for hammock sway.
        """
        # Constant forward velocity and yaw rate for circular path
        vx = self.forward_velocity
        yaw_rate = self.yaw_rate
        
        # Sinusoidal lateral velocity for hammock sway
        # 2.5 oscillations per full phase cycle [0,1]
        vy = self.lateral_sway_amplitude * np.sin(2 * np.pi * self.lateral_sway_frequency * phase)

        # Set velocities in world frame
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])

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
        Compute foot position in body frame with continuous contact.
        
        Feet maintain ground contact while adjusting position to compensate for:
        - Forward base motion (shuffle backward in body frame)
        - Lateral sway (adjust perpendicular to forward direction)
        - Yaw rotation (handled implicitly by body frame expression)
        
        Diagonal pairs (FL+RR, FR+RL) coordinate micro-adjustments for stability.
        """
        # Get leg-specific phase with diagonal coordination offset
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Backward shuffle in body frame to compensate for forward base velocity
        # Smooth sinusoidal motion for continuous contact
        shuffle_x = self.shuffle_amplitude_x * np.sin(2 * np.pi * leg_phase)
        foot[0] += shuffle_x
        
        # Lateral adjustment to compensate for sway
        # Counter-sway: when base moves right (+y), feet adjust left (-y) in body frame
        # Use higher frequency lateral oscillation matching the sway pattern
        lateral_adjust = -self.shuffle_amplitude_y * np.sin(2 * np.pi * self.lateral_sway_frequency * phase)
        
        # Apply differential lateral adjustment based on leg position
        # Front vs rear legs adjust slightly differently to maintain stance polygon
        if leg_name.startswith('F'):  # Front legs
            foot[1] += lateral_adjust * 0.8
        else:  # Rear legs
            foot[1] += lateral_adjust * 1.2
        
        # Left vs right legs have opposite lateral bias
        if leg_name.endswith('L'):
            foot[1] += 0.02 * np.cos(2 * np.pi * leg_phase)
        else:
            foot[1] -= 0.02 * np.cos(2 * np.pi * leg_phase)
        
        # Small vertical adjustment to maintain smooth ground contact
        # during compound motion (compensates for any body roll/pitch coupling)
        foot[2] += 0.005 * np.sin(4 * np.pi * leg_phase)
        
        return foot