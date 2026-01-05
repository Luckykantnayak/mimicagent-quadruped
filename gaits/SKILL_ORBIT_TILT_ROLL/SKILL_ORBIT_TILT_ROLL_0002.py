from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_ORBIT_TILT_ROLL_MotionGenerator(BaseMotionGenerator):
    """
    Orbit with synchronized roll oscillation.
    
    Robot travels in a continuous circular path while executing rhythmic roll oscillations.
    Base rolls left-neutral-right-neutral cyclically, with outer legs extending to support
    the tilt while maintaining forward orbital progression.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        # Initialize base class
        BaseMotionGenerator.__init__(self, initial_foot_positions_body, freq=0.5)
        
        self.leg_names = leg_names
        
        # Motion parameters
        self.freq = 0.5  # Complete one orbit + full roll cycle per period
        
        # Base motion parameters for circular trajectory
        self.forward_velocity = 0.8  # m/s forward velocity
        self.yaw_rate = 1.5  # rad/s to create circular path
        
        # Roll oscillation parameters
        self.max_roll_angle = np.deg2rad(20)  # Peak roll amplitude (20 degrees)
        
        # Leg extension parameters
        self.lateral_extension_max = 0.12  # Maximum lateral extension for outer legs (m)
        self.vertical_extension_max = 0.06  # Maximum vertical extension for outer legs (m)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Identify left and right legs
        self.left_legs = [leg for leg in leg_names if leg.startswith('FL') or leg.startswith('RL')]
        self.right_legs = [leg for leg in leg_names if leg.startswith('FR') or leg.startswith('RR')]

    def update_base_motion(self, phase, dt):
        """
        Update base motion: constant forward velocity with constant yaw rate for circular path,
        plus sinusoidal roll oscillation synchronized to phase.
        
        Roll profile:
        - phase [0, 0.25]: roll left (negative roll accumulates)
        - phase [0.25, 0.5]: transition through neutral to right roll
        - phase [0.5, 0.75]: roll right (positive roll accumulates)
        - phase [0.75, 1.0]: transition back to neutral
        """
        
        # Constant forward velocity and yaw rate for circular trajectory
        vx = self.forward_velocity
        yaw_rate = self.yaw_rate
        
        # Roll rate oscillates sinusoidally to produce smooth roll oscillation
        # Roll rate = d(roll)/dt = -max_roll * 2π * freq * sin(2π * freq * t)
        # This integrates to: roll(t) = max_roll * cos(2π * freq * t)
        # At phase=0: roll=max_roll (left), phase=0.5: roll=-max_roll (right)
        # We want: phase=0.25: peak left, phase=0.75: peak right
        # Adjust by phase shifting: roll(phase) = -max_roll * cos(2π * (phase - 0.25))
        #                                        = -max_roll * cos(2π * phase - π/2)
        #                                        = -max_roll * sin(2π * phase)
        # Roll rate = d(roll)/dphase * dphase/dt = -max_roll * 2π * cos(2π * phase) * freq
        
        roll_rate = -self.max_roll_angle * 2 * np.pi * self.freq * np.cos(2 * np.pi * phase)
        
        # Set velocities
        self.vel_world = np.array([vx, 0.0, 0.0])
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
        Compute foot position in body frame with lateral and vertical extensions
        based on roll phase.
        
        Outer legs (relative to roll direction) extend laterally and vertically.
        Inner legs retract toward body centerline.
        
        Roll profile:
        - phase [0, 0.25]: rolling left -> right legs are outer (extend right)
        - phase [0.25, 0.5]: transition -> extensions reverse
        - phase [0.5, 0.75]: rolling right -> left legs are outer (extend left)
        - phase [0.75, 1.0]: transition -> return to neutral
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left or right leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Compute extension factor based on phase and leg side
        # Extension factor: 1.0 = fully extended (outer leg), 0.0 = neutral/retracted (inner leg)
        
        if is_left_leg:
            # Left legs are outer during right roll (phase 0.5-0.75, peak at 0.75)
            # Use sinusoid: extension peaks at phase=0.625 (midpoint of right roll)
            # Extension profile: low at phase=0, increases through 0.25-0.75, peaks ~0.625, decreases to 1.0
            # Use: extension = 0.5 * (1 - cos(2π * (phase - 0.125)))
            # This gives: phase=0.125->0, phase=0.625->1, phase=1.125(=0.125)->0
            # Shift to: extension = 0.5 * (1 - cos(2π * (phase + 0.375)))
            extension_factor = 0.5 * (1.0 - np.cos(2 * np.pi * (phase + 0.375)))
            
        else:  # is_right_leg
            # Right legs are outer during left roll (phase 0.0-0.25, peak at 0.25)
            # Extension peaks at phase=0.125 (midpoint of left roll)
            # Use: extension = 0.5 * (1 + cos(2π * (phase + 0.375)))
            extension_factor = 0.5 * (1.0 + np.cos(2 * np.pi * (phase + 0.375)))
        
        # Apply lateral extension (y-direction in body frame)
        # Positive y = left, negative y = right
        lateral_sign = 1.0 if is_left_leg else -1.0
        foot[1] += lateral_sign * self.lateral_extension_max * extension_factor
        
        # Apply vertical extension (z-direction in body frame)
        # Negative z = down (extending legs downward)
        foot[2] -= self.vertical_extension_max * extension_factor
        
        return foot